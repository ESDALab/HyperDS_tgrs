import numpy as np
import torch
import torch.utils.data as data
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import logging
from dataset.builder import build_dataset
from models.builder import build_model

import os
import json
import glob
import time
from utils.time_metric import TimeMetric
import tqdm
import shutil
import torch.nn.functional as F
import sys
# sys.path.append('/mnt/petrelfs/liuzili/code/DA_RS')
from utils.positional_encoding import SineCosPE
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs
from src.vis_field import VisField
import xarray as xr

class DataAssimilationRemoteSensingDistGrid():
    def __init__(self, config):
        self.config = config
        self.with_gt = self.config.train_cfg.with_gt
        self.target_resolution = self.config.train_cfg.target_resolution
        self.with_stn = self.config.train_cfg.with_stn
        self.with_h8 = self.config.train_cfg.with_h8
        self.lon_range = self.config.train_cfg.lon_range
        self.lat_range = self.config.train_cfg.lat_range
        self.VisUtil = VisField(self.lon_range, self.lat_range)
        self.pred_name_list = self.config.train_cfg.pred_names.surface
        # self.device = config.device
        self.dx = self.config.train_cfg.dx
        self.dy = self.config.train_cfg.dy
        self.out_lon_size = 144
        self.out_lat_size = 224
        self.pe = SineCosPE(2, include_input=False)

        self.exp_path = os.path.join(self.config.exp_parent_path, self.config.exp_name)
        self.log_path = os.path.join(self.exp_path,'logs')
        os.makedirs(self.log_path, exist_ok=True)
        self.with_vis = self.config.train_cfg.log.with_vis
        if self.with_vis:
            self.vis_path = os.path.join(self.log_path, 'vis')
            os.makedirs(self.vis_path, exist_ok=True)
        # accelerator_project_config = ProjectConfiguration(project_dir=self.exp_path, logging_dir=self.log_path)

        self.accelerator = Accelerator(
                                    kwargs_handlers=[DistributedDataParallelKwargs(broadcast_buffers=False,find_unused_parameters=True)]
                                    # , find_unused_parameters=True
                                    )
        self.device = self.accelerator.device
        self._build()
    
    def _build(self):
        self._build_dir()
        self._build_data_loader()
        self._build_model()
        self._build_optimizer()

    def PoseEncoding(self, x, y, K):
        x = x / self.config.train_cfg.dx / self.out_lon_size
        y = y / self.config.train_cfg.dy / self.out_lat_size
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        
        if len(x.shape) == 1:
            sample_input = torch.stack([x,y], dim=1)
        else:
            sample_input = torch.cat([x,y], dim=1)
        
        pEn = []
        for k in range(K):
            w = np.pi / (2. ** (k+1))
            sinp = torch.sin(w * sample_input)
            cosp = torch.cos(w * sample_input)
            pEn.append(torch.cat([sinp, cosp], dim=1))
        pEn = torch.cat(pEn, dim=1)
        
        return pEn

    def interp_pred_grid(self, pred_grid, w2k_x, w2k_y):
        pred_grid = pred_grid.cpu().detach().numpy()
        w2k_x = w2k_x.cpu().detach().numpy()
        w2k_y = w2k_y.cpu().detach().numpy()
        _, _, H, W = pred_grid.shape
        in_lon = self.lon_range[0] + np.array(range(W)) * self.target_resolution
        in_lat = self.lat_range[0] + np.array(range(H)) * self.target_resolution
        in_lat = in_lat[::-1]
        
        p_lon = (w2k_x/self.dx)*self.target_resolution + self.lon_range[0]
        p_lat = (w2k_y/self.dy)*self.target_resolution + self.lat_range[0]
        interp_data = []
        for channel in range(pred_grid.shape[1]):
            coord_x = in_lon
            coord_y = in_lat
            data = xr.DataArray(data=pred_grid[:,channel,:,:], dims=['batch', 'y','x'],
                                coords=(np.arange(pred_grid.shape[0]), coord_y.tolist(), coord_x.tolist()))
            var_list = data.interp(x=xr.DataArray(p_lon[0,:], dims='z'),
                                   y=xr.DataArray(p_lat[0,:], dims='z'))
            interp_data.append(var_list)
        interp_data = np.stack(interp_data, axis=-1)
        interp_data = torch.from_numpy(interp_data).float()
        return interp_data
            
    
    def train_one_epoch(self, epoch):
        if self.with_gt:
            train_loss_dict = {'baseline_grid_loss': 0,
                               'pred_grid_loss': 0,
                               'baseline_station_loss': 0,
                               'pred_station_loss': 0
                               }
        else:
            train_loss_dict = {'baseline_grid_loss': 0,
                               'pred_grid_loss': 0,
                               'baseline_station_loss': 0,
                               'pred_station_loss': 0,
                               }
        for i, data in enumerate(self.train_dataloader):
                if i >= 2 and (self.glob_step-1) % self.log_step == 0:
                    end_time = time.time()
                    iter_time = (end_time - start_time) 
                    self.accelerator.print(f"[Epoch:{epoch}/{self.num_epoch}][batch:{i}/{len(self.train_dataloader)}]: loss_baseline:{loss_baseline.item()}/gird_loss:{train_grid_loss.item()}/stn_interp_loss:{interp_stn_loss.item()}/train_stn_loss:{train_stn_loss.item()}/speed{iter_time}s")
                start_time = time.time()
                
                self.glob_step = self.glob_step + 1
                self.model.train()
                
                h8_data = data['h8_data'].to(self.device)
                if not self.with_h8:
                    h8_data = torch.zeros(h8_data.shape).to(self.device)
                era5_data_surface_input = data['era5_data_surface_input'][:,0,:,:,:].to(self.device)
                era5_data_surface_label = data['era5_data_surface_label'][:,0,:,:,:].to(self.device)
                
                
                # grid supervision
                pred = self.model.forward(h8_data, era5_data_surface_input)
                
                # cal mean value for each pixel and add shortcut
                H, W = pred.shape[-2:]
                era5_data_surface_interp = nn.functional.interpolate(era5_data_surface_input, align_corners=True, size=[H, W], mode='bilinear')
                pred = pred + era5_data_surface_interp
                
                loss_baseline = self.grid_criterion(era5_data_surface_interp, era5_data_surface_label)
                train_loss_dict['baseline_grid_loss'] += loss_baseline
                
                
                if self.with_gt:
                    loss_pred_gt = self.grid_criterion(pred, era5_data_surface_label)
                    train_loss_dict['pred_grid_loss'] += loss_pred_gt
                    train_grid_loss = loss_pred_gt
                else:
                    loss_pred_interp = self.grid_criterion(pred, era5_data_surface_interp)
                    train_loss_dict['pred_grid_loss'] += loss_pred_interp
                    train_grid_loss = loss_pred_interp
                
                
                # station supervision
                
                w2k_x = data['w2k_x'].to(self.device)
                w2k_y = data['w2k_y'].to(self.device)
                w2k_data = data['w2k_interp'].to(self.device)
                w2k_label = data['w2k_label'].to(self.device)
                pred_interp = self.interp_pred_grid(pred, w2k_x, w2k_y).to(self.device)
                
                pred_wind = torch.sqrt(pred_interp[:,:,0]**2 + pred_interp[:,:,1]**2)
                
                w2k_loss_wind = self.stn_criterion(w2k_label[:,:,0], pred_wind)
                w2k_loss_ot = self.stn_criterion(w2k_label[:,:,1:], pred_interp[:,:,2:])

                w2k_interp_wind = torch.sqrt(w2k_data[:,:,0]**2 + w2k_data[:,:,1]**2)
                w2k_interp_ot = w2k_data[:,:,2:]

                w2k_interp_loss_wind = self.stn_criterion(w2k_label[:,:,0], w2k_interp_wind)
                w2k_interp_loss_ot = self.stn_criterion(w2k_label[:,:,1:], w2k_interp_ot)

                train_stn_loss = w2k_loss_ot + w2k_loss_wind
                train_loss_dict['pred_station_loss'] += train_stn_loss
                interp_stn_loss = w2k_interp_loss_ot + w2k_interp_loss_wind
                train_loss_dict['baseline_station_loss'] += interp_stn_loss

                if self.with_stn:
                    train_loss = train_grid_loss + 0.05*train_stn_loss
                else:
                    train_loss = train_grid_loss + 0.*train_stn_loss
                
                
                if self.accelerator.is_main_process:
                    self.log_writer.add_scalars(main_tag='training/train_gird_loss', 
                                                tag_scalar_dict={'loss_pred':train_grid_loss.item(),
                                                                'loss_interp_baseline':loss_baseline.item()}, 
                                                global_step=self.glob_step)
                    
                    self.log_writer.add_scalars(main_tag='training/train_stn_loss',
                                                tag_scalar_dict={'loss_pred':train_stn_loss.item(),
                                                                'loss_interp_baseline':interp_stn_loss.item()},
                                                global_step= self.glob_step)
            
                self.optimizer.zero_grad()
                self.accelerator.backward(train_loss)
                self.optimizer.step()
                
                if (self.glob_step-1) % self.log_step*10 == 0:

                    # self.accelerator.print(f"[Epoch:{epoch}/{self.num_epoch}][batch:{i}/{len(self.train_dataloader)}]: loss_baseline:{loss_baseline}/gird_loss:{loss_pred_gt}/stn_interp_loss:{interp_stn_loss}/train_stn_loss:{train_stn_loss}")

                    if self.accelerator.is_main_process:
                        if self.with_vis:
                            pred_inver = self.inverse_norm(pred)
                            era5_input_inver = self.inverse_norm(era5_data_surface_input)
                            era5_gt_inver = self.inverse_norm(era5_data_surface_label)
                            era5_interp_inver = self.inverse_norm(era5_data_surface_interp)
                            for id, var_name in enumerate(self.pred_name_list):
                                result_file_path = os.path.join(self.vis_path, 'train_vis')
                                os.makedirs(result_file_path, exist_ok=True)
                                result_file = os.path.join(result_file_path, f'{self.glob_step}_{var_name}.png')
                                self.VisUtil.forward(era5_input_inver[0, id, :, :].numpy(), pred_inver[0, id, :, :].numpy(),
                                                    era5_interp_inver[0, id, :, :].numpy(), era5_gt_inver[0, id, :, :].numpy(), var_name, result_file)
        
        return train_loss_dict
    
    def valid_one_epoch(self, epoch):
        if self.with_gt:
            valid_loss_dict = {'baseline_grid_loss': 0,
                               'pred_grid_loss': 0,
                               'baseline_station_loss': 0,
                               'pred_station_loss': 0
                               }
        else:
            valid_loss_dict = {'baseline_grid_loss': 0,
                               'pred_grid_loss': 0,
                               'baseline_station_loss': 0,
                               'pred_station_loss': 0,
                               }
        with torch.no_grad():
            for i, data in enumerate(self.valid_dataloader):
                h8_data = data['h8_data'].to(self.device)
                if not self.with_h8:
                    h8_data = torch.zeros(h8_data.shape).to(self.device)
                era5_data_surface_input = data['era5_data_surface_input'][:,0,:,:,:].to(self.device)
                era5_data_surface_label = data['era5_data_surface_label'][:,0,:,:,:].to(self.device)
                
                
                # grid supervision
                pred = self.model.forward(h8_data, era5_data_surface_input)

                
                # cal mean value for each pixel and add shortcut
                H, W = pred.shape[-2:]
                era5_data_surface_interp = nn.functional.interpolate(era5_data_surface_input, align_corners=True, size=[H, W], mode='bilinear')
                pred = pred + era5_data_surface_interp
                
                loss_baseline = self.grid_criterion(era5_data_surface_interp, era5_data_surface_label)
                valid_loss_dict['baseline_grid_loss'] += loss_baseline


                if self.with_gt:
                    loss_pred_gt = self.grid_criterion(pred, era5_data_surface_label)
                    valid_loss_dict['pred_grid_loss'] += loss_pred_gt
                else:
                    loss_pred_interp = self.grid_criterion(pred, era5_data_surface_interp)
                    valid_loss_dict['pred_grid_loss'] += loss_pred_interp

                # station supervision
            
                w2k_x = data['w2k_x'].to(self.device)
                w2k_y = data['w2k_y'].to(self.device)
                w2k_data = data['w2k_interp'].to(self.device)
                w2k_label = data['w2k_label'].to(self.device)
                pred_interp = self.interp_pred_grid(pred, w2k_x, w2k_y).to(self.device)
                
                pred_wind = torch.sqrt(pred_interp[:,:,0]**2 + pred_interp[:,:,1]**2)
                
                w2k_loss_wind = self.stn_criterion(w2k_label[:,:,0], pred_wind)
                w2k_loss_ot = self.stn_criterion(w2k_label[:,:,1:], pred_interp[:,:,2:])

                w2k_interp_wind = torch.sqrt(w2k_data[:,:,0]**2 + w2k_data[:,:,1]**2)
                w2k_interp_ot = w2k_data[:,:,2:]

                w2k_interp_loss_wind = self.stn_criterion(w2k_label[:,:,0], w2k_interp_wind)
                w2k_interp_loss_ot = self.stn_criterion(w2k_label[:,:,1:], w2k_interp_ot)

                valid_stn_loss = w2k_loss_ot + w2k_loss_wind
                valid_loss_dict['pred_station_loss'] += valid_stn_loss
                interp_stn_loss = w2k_interp_loss_ot + w2k_interp_loss_wind
                valid_loss_dict['baseline_station_loss'] += interp_stn_loss


            '''
            # if self.accelerator.is_main_process:
            #     self.log_writer.add_scalars(main_tag='validating/valid_gird_loss', 
            #                                 tag_scalar_dict={'loss_pred':valid_grid_loss,
            #                                                 'loss_interp_baseline':loss_baseline}, 
            #                                 global_step=self.glob_step)
            
            #     self.log_writer.add_scalars(main_tag='validating/valid_stn_loss',
            #                                 tag_scalar_dict={'loss_pred':valid_stn_loss,
            #                                                 'loss_interp_baseline':interp_stn_loss},
            #                                         global_step= self.glob_step)
            # self.accelerator.print(f"valid: loss_baseline:{loss_baseline}/gird_loss:{loss_pred_gt}/stn_interp_loss:{interp_stn_loss}/valid_stn_loss:{valid_stn_loss}")
            '''
        return valid_loss_dict
    
    def train(self):
        self.batch_size = self.config.train_cfg.batch_size
        self.num_epoch = self.config.train_cfg.num_epoch
        save_epoch = self.config.train_cfg.checkpoint.save_epoch
        self.log_step = self.config.train_cfg.log.log_step
        self.pe.to(self.device)
        self.model.to(self.device)

        self.grid_criterion = self._build_loss()
        self.stn_criterion = self._build_loss()

        lr = self.optimizer.param_groups[0]['lr']
        
        self.accelerator.print(f"Set lr to: {lr}")
        self.model, self.optimizer, self.train_dataloader = self.accelerator.prepare(self.model, self.optimizer, self.train_dataloader)

        for epoch in range(self.epoch, self.num_epoch):
            self.model.train()
            train_loss_dict = self.train_one_epoch(epoch)
            # self.train_one_epoch(epoch)
            self.model.eval()
            val_loss_dict = self.valid_one_epoch(epoch)
                    
            self.accelerator.print(f"=>[Epoch:{epoch}/{self.num_epoch}]")
            self.accelerator.print(f"==>[train_loss]:\nbaseline_pred_loss:{train_loss_dict['baseline_grid_loss']/len(self.train_dataloader)}\npred_grid_loss:{train_loss_dict['pred_grid_loss']/len(self.train_dataloader)}\nbaseline_station_loss:{train_loss_dict['baseline_station_loss']/len(self.train_dataloader)}\npred_station_loss:{train_loss_dict['pred_station_loss']/len(self.train_dataloader)}")
            self.accelerator.print(f"==>[valid_loss]:\nbaseline_pred_loss:{val_loss_dict['baseline_grid_loss']/len(self.valid_dataloader)}\npred_grid_loss:{val_loss_dict['pred_grid_loss']/len(self.valid_dataloader)}\nbaseline_station_loss:{val_loss_dict['baseline_station_loss']/len(self.valid_dataloader)}\npred_station_loss:{val_loss_dict['pred_station_loss']/len(self.valid_dataloader)}")
            
            if self.accelerator.is_main_process:
                self.log_writer.add_scalars(main_tag='training_epoch/train_gird_loss', 
                                            tag_scalar_dict={'loss_pred':train_loss_dict['pred_grid_loss']/len(self.train_dataloader),
                                                            'loss_interp_baseline':train_loss_dict['baseline_grid_loss']/len(self.train_dataloader)}, 
                                            global_step=epoch)
                
                self.log_writer.add_scalars(main_tag='training_epoch/train_stn_loss',
                                            tag_scalar_dict={'loss_pred':train_loss_dict['pred_station_loss']/len(self.train_dataloader),
                                                            'loss_interp_baseline':train_loss_dict['baseline_station_loss']/len(self.train_dataloader)},
                                            global_step=epoch)     

                self.log_writer.add_scalars(main_tag='valid_epoch/valid_gird_loss', 
                                            tag_scalar_dict={'loss_pred':val_loss_dict['pred_grid_loss']/len(self.valid_dataloader),
                                                            'loss_interp_baseline':val_loss_dict['baseline_grid_loss']/len(self.valid_dataloader)}, 
                                            global_step=epoch)
                
                self.log_writer.add_scalars(main_tag='valid_epoch/valid_stn_loss',
                                            tag_scalar_dict={'loss_pred':val_loss_dict['pred_station_loss']/len(self.valid_dataloader),
                                                            'loss_interp_baseline':val_loss_dict['baseline_station_loss']/len(self.valid_dataloader)},
                                            global_step=epoch)               

                if 'lr_schedule' in self.config.train_cfg.keys():
                    self.lr_schedule.step()
                    
                if epoch % save_epoch == 0:
                    lr = self.optimizer.param_groups[0]['lr']
                    if self.accelerator.is_main_process:
                        self.log_writer.add_scalar('learning_rate', lr, self.glob_step)
                    state_dict = {}
                    if isinstance(self.model, nn.Module):
                        state_dict['model'] = self.model.state_dict()
                    else:
                        state_dict['model'] = self.model
                    state_dict['epoch'] = epoch
                    state_dict['global_step'] = self.glob_step

                    model_file = os.path.join(self.checkpoint_path, f"{self.config.model_name}_epoch{epoch}.pth")
                    unwarpped_model = self.accelerator.unwrap_model(self.model)
                    state_dict['model'] = unwarpped_model
                    self.accelerator.save(state_dict, model_file)
                    shutil.copy(model_file, os.path.join(self.checkpoint_path, 'latest.pth'))
            
    def inference(self):
        self.model.eval()
        self.pe.to(self.device)
        self.model.to(self.device)
        self.grid_criterion = self._build_loss()
        self.stn_criterion = self._build_loss()
        if self.with_gt:
            test_loss_dict = {'baseline_grid_loss': 0,
                               'pred_grid_loss': 0,
                               'baseline_station_loss': 0,
                               'pred_station_loss': 0
                               }
            w2k_matric_dict = {'mse': {
                                        'wind': 0,
                                        't2m': 0,
                                        'sp': 0,
                                        'tp1h': 0
                                        },
                               'mae': {'wind': 0,
                                        't2m': 0,
                                        'sp': 0,
                                        'tp1h': 0
                                        }
                               }
        else:
            test_loss_dict = {'baseline_grid_loss': 0,
                               'pred_grid_mean_loss': 0,
                               'pred_grid_interp_loss': 0,
                               'baseline_station_loss': 0,
                               'pred_station_loss': 0,
                               }
            w2k_matric_dict = {'mse': {
                                        'wind': 0,
                                        't2m': 0,
                                        'sp': 0,
                                        'tp1h': 0
                                        },
                               'mae': {'wind': 0,
                                        't2m': 0,
                                        'sp': 0,
                                        'tp1h': 0
                                        }
                               }
        dataset_len = 0
        with torch.no_grad():
            progress_bar = tqdm.tqdm(enumerate(self.test_dataloader), total=len(self.test_dataloader))  
            for i,data in progress_bar:
                dataset_len += 1
                h8_data = data['h8_data'].to(self.device)
                BSize = h8_data.shape[0]
                if not self.with_h8:
                    h8_data = torch.zeros(h8_data.shape).to(self.device)
                era5_data_surface_input = data['era5_data_surface_input'][:,0,:,:,:].to(self.device)
                era5_data_surface_label = data['era5_data_surface_label'][:,0,:,:,:].to(self.device)
                pred = self.model.forward(h8_data, era5_data_surface_input)

                # cal mean value for each pixel and add shortcut
                H, W = pred.shape[-2:]
                era5_data_surface_interp = nn.functional.interpolate(era5_data_surface_input, align_corners=True, size=[H, W], mode='bilinear')
                pred = pred + era5_data_surface_interp
                
                pred_inver = self.inverse_norm(pred)

                loss_baseline = self.grid_criterion(era5_data_surface_interp, era5_data_surface_label)
                test_loss_dict['baseline_grid_loss'] += loss_baseline

                if self.with_gt:
                    loss_pred_gt = self.grid_criterion(pred, era5_data_surface_label)
                    test_loss_dict['pred_grid_loss'] += loss_pred_gt
                else:
                    H, W = era5_data_surface_input.shape[-2:]
                    pred_mean = F.adaptive_avg_pool2d(pred, [H, W])
                    loss_surface = self.grid_criterion(pred_mean, era5_data_surface_input)
                    test_loss_dict['pred_grid_mean_loss'] += loss_surface

                    loss_pred_interp = self.grid_criterion(pred, era5_data_surface_interp)
                    test_loss_dict['pred_grid_interp_loss'] += loss_pred_interp
                
                w2k_x = data['w2k_x'].to(self.device)
                w2k_y = data['w2k_y'].to(self.device)
                w2k_data = data['w2k_interp'].to(self.device)
                w2k_label = data['w2k_label'].to(self.device)
                interp_pred = self.interp_pred_grid(pred, w2k_x, w2k_y).to(self.device)  
                interp_pred_1 = interp_pred.permute(0,2,1).contiguous()
                pred_wind = torch.sqrt(interp_pred_1[:,0,:]**2 + interp_pred_1[:,1,:]**2) 
                
                tmp_label = self.inverse_norm(w2k_label[:,:,0],data_type='w2k',var_name='wind')
                tmp_pred = self.inverse_norm(pred_wind,data_type='w2k',var_name='wind')
                w2k_mse_wind = F.mse_loss(torch.from_numpy(tmp_label),torch.from_numpy(tmp_pred),reduction='mean')
                w2k_mae_wind = F.l1_loss(torch.from_numpy(tmp_label),torch.from_numpy(tmp_pred),reduction='mean')
                w2k_matric_dict['mse']['wind'] += w2k_mse_wind
                w2k_matric_dict['mae']['wind'] += w2k_mae_wind
                tmp_label = self.inverse_norm(w2k_label.permute(0,2,1)[:,1,:].contiguous(),data_type='w2k',var_name='t2m')
                tmp_pred = self.inverse_norm(interp_pred_1[:,2,:],data_type='w2k',var_name='t2m')
                
                w2k_mse_t2m = F.mse_loss(torch.from_numpy(tmp_label),torch.from_numpy(tmp_pred),reduction='mean')
                w2k_mae_t2m = F.l1_loss(torch.from_numpy(tmp_label),torch.from_numpy(tmp_pred),reduction='mean')
                w2k_matric_dict['mse']['t2m'] += w2k_mse_t2m
                w2k_matric_dict['mae']['t2m'] += w2k_mae_t2m
                tmp_label = self.inverse_norm(w2k_label.permute(0,2,1)[:,2,:].contiguous(),data_type='w2k',var_name='sp')
                tmp_pred = self.inverse_norm(interp_pred_1[:,3,:],data_type='w2k',var_name='sp')
                w2k_mse_sp = F.mse_loss(torch.from_numpy(tmp_label)/100,torch.from_numpy(tmp_pred)/100,reduction='mean')
                w2k_mae_sp = F.l1_loss(torch.from_numpy(tmp_label)/100,torch.from_numpy(tmp_pred)/100,reduction='mean')
                w2k_matric_dict['mse']['sp'] += w2k_mse_sp
                w2k_matric_dict['mae']['sp'] += w2k_mae_sp
                tmp_label = self.inverse_norm(w2k_label.permute(0,2,1)[:,3,:].contiguous(),data_type='w2k',var_name='tp1h')
                tmp_pred = self.inverse_norm(interp_pred_1[:,4,:],data_type='w2k',var_name='tp1h')
                w2k_mse_tp1h = F.mse_loss(torch.from_numpy(tmp_label),torch.from_numpy(tmp_pred),reduction='mean')
                w2k_mae_tp1h = F.l1_loss(torch.from_numpy(tmp_label),torch.from_numpy(tmp_pred),reduction='mean')
                w2k_matric_dict['mse']['tp1h'] += w2k_mse_tp1h
                w2k_matric_dict['mae']['tp1h'] += w2k_mae_tp1h
                
                if i % 200 == 0:
                    vis_stn_loss_dict = {
                        'wind': [],
                        't2m': [],
                        'sp': [],
                        'tp1h': []
                        }
                    for stn_id in range(interp_pred.shape[1]):
                        pred_wind = torch.sqrt(interp_pred[:,stn_id,0]**2 + interp_pred[:,stn_id,1]**2)
                        wind_loss_tmp = F.mse_loss(pred_wind, w2k_label[:,stn_id,0])
                        vis_stn_loss_dict['wind'].append(wind_loss_tmp.item())
                        
                        t2m_loss_tmp = F.mse_loss(interp_pred[:,stn_id,2], w2k_label[:,stn_id,1])
                        vis_stn_loss_dict['t2m'].append(t2m_loss_tmp.item())

                        sp_loss_tmp = F.mse_loss(interp_pred[:,stn_id,3], w2k_label[:,stn_id,2])
                        vis_stn_loss_dict['sp'].append(sp_loss_tmp.item())

                        tp1h_loss_tmp = F.mse_loss(interp_pred[:,stn_id,4], w2k_label[:,stn_id,3])
                        vis_stn_loss_dict['tp1h'].append(tp1h_loss_tmp.item())
                    
                    for id, var_name in enumerate(self.config.train_cfg.pred_names.surface):
                        result_file_path = os.path.join(self.vis_path, 'test_vis')
                        os.makedirs(result_file_path, exist_ok=True)
                        if var_name in ['u10', 'v10']:
                            result_file = os.path.join(result_file_path, f'step{i}_wind.png')
                            pred_inver_ws = torch.sqrt(pred_inver[0,0,:,:]**2+pred_inver[0,1,:,:]**2)
                            self.VisUtil.forward_single_image_w_stn(pred_inver_ws, vis_stn_loss_dict['wind'], w2k_x, w2k_y, var_name, result_file)
                        else:
                            result_file = os.path.join(result_file_path, f'step{i}_{var_name}.png')
                            self.VisUtil.forward_single_image_w_stn(pred_inver[0, id, :, :], vis_stn_loss_dict[var_name], w2k_x, w2k_y, var_name, result_file)
            #print(f"==>[test_loss]:\nbaseline_station_loss:{test_loss_dict['baseline_station_loss']/len(dataloader)}\npred_station_loss:{test_loss_dict['pred_station_loss']/len(dataloader)}")
            self.accelerator.print(f"==>[test_loss]:{w2k_matric_dict},data_len:{i}")  
                        
    def _build_dir(self):
        # logger
        
        if self.accelerator.is_main_process:
            log_name = f"{time.strftime('%Y-%m-%d-%H-%M')}.log"
            log_name = f"{self.config.exp_name}_{log_name}"
            log_dir = os.path.join(self.log_path, log_name)
            self.log = logging.getLogger()
            self.log.setLevel(logging.INFO)
            handler = logging.FileHandler(log_dir)
            self.log.addHandler(handler)
            self.log_writer = SummaryWriter(log_dir= self.log_path)

            self.log.info("Config:")
            self.log.info(self.config)
            self.log.info("\n")
        self.accelerator.print("Config:", self.config)

        self.pred_data_path = self.config.train_cfg.pred_data_path
        self.h8_data_path = self.config.train_cfg.h8_data_path

        self.checkpoint_path = os.path.join(self.config.exp_parent_path, self.config.exp_name+'/checkpoint')
        os.makedirs(self.checkpoint_path, exist_ok=True)
    
    def _build_data_loader(self):
        self.accelerator.print("===> Loading dataloader......")
        if self.config.mode == 'train':
            self.train_dataset = build_dataset(self.config.dataset_name, self.config.train_cfg, 'train')

            self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size = self.config.train_cfg.batch_size, shuffle=True,
                                                                drop_last=True,num_workers=self.config.train_cfg.num_workers,
                                                                pin_memory = True, prefetch_factor = 3, persistent_workers = True)

            self.valid_dataset = build_dataset(self.config.dataset_name, self.config.train_cfg, 'valid')

            self.valid_dataloader = torch.utils.data.DataLoader(self.valid_dataset, batch_size = self.config.train_cfg.batch_size, shuffle=False,
                                                                drop_last=True,num_workers=self.config.train_cfg.num_workers,
                                                                pin_memory = True, prefetch_factor = 3, persistent_workers = True)
        elif self.config.mode == 'test':
            self.test_dataset = build_dataset(self.config.dataset_name, self.config.train_cfg, 'test')
            self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size = self.config.train_cfg.batch_size, shuffle=False,
                                                                drop_last=True,num_workers=self.config.train_cfg.num_workers,
                                                                pin_memory = True, prefetch_factor = 3, persistent_workers = True)
        self.accelerator.print("===> Loaded dataloader !")

    def _build_model(self):
        self.accelerator.print("===> Initializing Model......")
        self.model = build_model(self.config.model_name, self.config.network)
        if self.config.train_cfg.resume or self.config.test_cfg.resume:
            if self.config.mode == 'train':
                checkpoint_cfg = self.config.train_cfg.checkpoint
            elif self.config.mode == 'test':
                checkpoint_cfg = self.config.test_cfg
            if checkpoint_cfg.checkpoint_name is None:
                model_file = os.path.join(self.checkpoint_path, checkpoint_cfg.checkpoint_name)
            else:
                # model_file = os.path.join(self.checkpoint_path, f"{self.config.exp_name}_latest.pth")
                model_file = checkpoint_cfg.checkpoint_name
            
            if not os.path.exists(model_file):
                self.accelerator.print(f"Warning: resume file {model_file} does not exist!")
                self.epoch, self.glob_step = 0, 0
            else:
                self.accelerator.print(f"Start to resume from {model_file}")
                state_dict = torch.load(model_file)
                try:
                    self.glob_step = state_dict.pop('global_step')
                except KeyError:
                    self.accelerator.print("Warning: global_step not in state dict!")
                    self.glob_step = 0
                try:
                    self.epoch = state_dict.pop('epoch')
                except KeyError:
                    self.accelerator.print("Warning: epoch not in state dict!")
                    self.epoch = 0
                self.accelerator.print(f"===> Resume form epoch {self.epoch} global step {self.glob_step} model file {model_file}")
                if 'model' in state_dict.keys():
                    self.model.load_state_dict(state_dict['model'].state_dict(), strict=True)

                    # self.model.load_state_dict(state_dict['model'], strict=True)
                else:
                    self.model.load_state_dict(state_dict, state_dict=True)
        else:
            self.accelerator.print("===> Initialized model, training model from scratch!")
            self.epoch = 0
            self.glob_step =0
                                                                                  
    def _build_optimizer(self):
        optim_dict = {'Adam': optim.Adam,
                      'SGD': optim.SGD}
        lr_schedule_dict = {'stepLR': optim.lr_scheduler.StepLR,
                            'CosineAnnealingLR': optim.lr_scheduler.CosineAnnealingLR}
        
        if self.config.train_cfg.optimizer.name in optim_dict.keys():
            self.optimizer = optim_dict[self.config.train_cfg.optimizer.pop('name')]([{'params': self.model.parameters(),
                                                                                **self.config.train_cfg.optimizer,
                                                                                'initial_lr': self.config.train_cfg.optimizer.lr}])
        else:
            raise NotImplementedError(f'Optimizer name {self.config.train_cfg.optimizer.name} not in optim_dict')
        
        if self.config.train_cfg.lr_schedule.name in lr_schedule_dict.keys():
            self.lr_schedule = lr_schedule_dict[self.config.train_cfg.lr_schedule.pop('name')](optimizer = self.optimizer,
                                                                                                  **self.config.train_cfg.lr_schedule, 
                                                                                                  last_epoch = self.epoch - 1)
        else:
            raise NotImplementedError(f'lr_schedule name {self.config.train_cfg.lr_schedule.name} not in lr_schedule_dict')
        
    def _build_loss(self):
        losses_dict = {'CrossEntropyLoss': nn.CrossEntropyLoss,
                       'L1Loss': nn.L1Loss,
                       'MSELoss': nn.MSELoss,
                       }
        loss_name = self.config.train_cfg.loss_cfg.name
        if 'scale' in self.config.train_cfg.loss_cfg.keys():
            self.scale = self.config.train_cfg.loss_cfg.scale
        if loss_name in losses_dict.keys():
            return losses_dict[loss_name]()
        else:
            raise NotImplementedError(f'Name {loss_name} not in losses_dcit')
        
    def inverse_norm(self, pred, prefix='single', data_type='grid', var_name='wind'):
        if self.device == 'cpu':
            pred = pred.detach().numpy()
        else:
            pred = pred.detach().cpu().numpy()
        if data_type.lower() == 'grid':
            if prefix == 'single':
                norm_file = self.config.train_cfg.norm_path.pred_single_norm_path
                with open(norm_file, mode='r') as f:
                    single_level_mean_std = json.load(f)
                if pred.shape[1] == 5:
                    for idx, var_name in enumerate(self.config.train_cfg.pred_names.surface):
                        mean = np.array(single_level_mean_std['mean'][var_name])
                        std = np.array(single_level_mean_std['std'][var_name])
                        pred[:,idx,:,:] = pred[:,idx,:,:] * std + mean
                    return torch.from_numpy(pred)
            else:
                raise NotImplementedError
        elif data_type.lower() == 'w2k':
            norm_file = self.config.train_cfg.norm_path.pred_single_norm_path
            with open(norm_file, mode='r') as f:
                single_level_mean_std = json.load(f)
            
            if var_name == 'wind':
                mean_wind = np.sqrt(single_level_mean_std['mean']['u10']**2+single_level_mean_std['mean']['v10']**2)
                std_wind = np.sqrt(single_level_mean_std['std']['u10']**2+single_level_mean_std['std']['v10']**2)
                
                return pred*std_wind + mean_wind
            else:
                mean = single_level_mean_std['mean'][var_name]
                std = single_level_mean_std['std'][var_name]
                return pred*std + mean
        