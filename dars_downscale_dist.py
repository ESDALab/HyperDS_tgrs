import numpy as np
import torch
import torch.utils.data as data
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import logging
from dataset.builder import build_dataset
from models.builder import build_model
import xarray as xr
import os
import json
import glob
import time
from utils.time_metric import TimeMetric
import tqdm
import shutil
import torch.nn.functional as F
import sys
from utils.positional_encoding import SineCosPE
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs
from src.vis_field import VisField

class DataAssimilationRemoteSensingDist():
    def __init__(self, config):
        self.config = config
        self.with_gt = self.config.train_cfg.with_gt
        self.with_stn = self.config.train_cfg.with_stn
        self.with_h8 = self.config.train_cfg.with_h8
        self.sample_mode = self.config.train_cfg.sample_mode
        self.lon_range = self.config.train_cfg.lon_range
        self.lat_range = self.config.train_cfg.lat_range
        self.input_resolution = self.config.train_cfg.input_resolution
        self.target_resolution = self.config.train_cfg.target_resolution
        self.VisUtil = VisField(self.lon_range, self.lat_range)
        self.sample_num = self.config.train_cfg.margin_sample_num
        self.pred_name_list = self.config.train_cfg.pred_names.surface
        self.decoder_type = config.network.decoder_type
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
                                    )
        self.device = self.accelerator.device
        self.has_normed = False
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

    def encoding_coord(self, x, y):
        '''
        x,y shape: batch x N
        reture sample_input.shape: batch x N x 128
        '''
        x = x / self.config.train_cfg.dx / (self.out_lon_size-1)
        y = y / self.config.train_cfg.dy / (self.out_lat_size-1)
        
        x = x.unsqueeze(2)
        y = y.unsqueeze(2)
        
        sample_input = torch.cat([x,y], dim=2)
        sample_input = self.pe.forward(sample_input)
        return sample_input

    def get_margin_grid(self, era5_data, h8_data):
        _, _, _, H, W = h8_data.shape
        BSize, var_num, LH, LW = era5_data.shape

        px, py = np.meshgrid(np.arange(W-1),np.arange(H-1))
        num_samples = self.sample_num
        px = np.repeat(px[:,:,np.newaxis], num_samples, axis=2) + np.random.rand(H-1,W-1,num_samples)
        py = np.repeat(py[:,:,np.newaxis], num_samples, axis=2) + np.random.rand(H-1,W-1,num_samples)
        px = px.ravel().tolist()
        py = py.ravel().tolist()
        
        p_lon = self.lon_range[0] + np.array(px) * self.target_resolution
        p_lat = self.lat_range[0] + np.array(py) * self.target_resolution
        p_lat = p_lat[::-1]
        in_lon = self.lon_range[0] + np.array(range(LW)) * self.input_resolution
        in_lat = self.lat_range[0] + np.array(range(LH)) * self.input_resolution
        in_lat = in_lat[::-1]
        
        coord_x = in_lon
        coord_y = in_lat
        data = xr.DataArray(data=era5_data.cpu().numpy(), dims=['bs','var_num','y','x'], coords=(np.array(range(BSize)).tolist(), np.array(range(var_num)).tolist(), coord_y.tolist(), coord_x.tolist()))
        var_list = data.interp(x=xr.DataArray(p_lon, dims='z'),
                                y=xr.DataArray(p_lat, dims='z'))
        
        margin_x = np.array(px) * self.dx
        margin_x = np.expand_dims(margin_x,0).repeat(BSize,axis=0)
        margin_y = np.array(py) * self.dy
        margin_y = np.expand_dims(margin_y,0).repeat(BSize,axis=0)
        del px,py,p_lon,p_lat,in_lon,in_lat
        inp_pred_surface_data = torch.from_numpy(np.array(var_list)).permute(0,2,1).contiguous().float()
        margin_x = torch.from_numpy(margin_x).float()
        margin_y = torch.from_numpy(margin_y).float()
        
        return margin_x, margin_y, inp_pred_surface_data
    

        w2k_item = w2k_item.cpu().numpy()
        BSize, var_num, LH, LW = era5_data.shape
        in_lon = self.lon_range[0] + np.array(range(LW)) * self.input_resolution
        in_lat = self.lat_range[0] + np.array(range(LH)) * self.input_resolution
        
        p_lon = w2k_item[0,:,1]
        p_lat = w2k_item[0,:,0]
        px = (p_lon-self.lon_range[0])/self.target_resolution
        py = (p_lat-self.lat_range[0])/self.target_resolution
        
        inp_pred_surface_data = []
        
        coord_x = in_lon
        coord_y = in_lat
        data = xr.DataArray(data=era5_data.cpu().numpy(), dims=['bs','var_num','y','x'], coords=(np.array(range(BSize)).tolist(), np.array(range(var_num)).tolist(), coord_y.tolist(), coord_x.tolist()))
        var_list = data.interp(x=xr.DataArray(p_lon, dims='z'),
                                y=xr.DataArray(p_lat, dims='z'))
        
        inp_pred_surface_data = np.array(var_list)
        w2k_label = []
        
        w2k_label.append(w2k_item[:, :, 16])
        w2k_label.append(w2k_item[:, :, 4]+273.15)
        w2k_label.append(w2k_item[:, :, 3]*100)
        w2k_label.append(w2k_item[:, :, 10])
        w2k_label = np.stack(w2k_label, axis=-1)
        
        w2k_x = np.array(px) * self.dx
        w2k_x = np.expand_dims(w2k_x,0).repeat(BSize,axis=0)
        w2k_y = np.array(py) * self.dy
        w2k_y = np.expand_dims(w2k_y,0).repeat(BSize,axis=0)
        
        inp_pred_surface_data = torch.from_numpy(inp_pred_surface_data).permute(0,2,1).contiguous().float()
        w2k_label = torch.from_numpy(w2k_label).float()
        w2k_x = torch.from_numpy(w2k_x).float()
        w2k_y = torch.from_numpy(w2k_y).float()
        if (not self.has_normed):
            w2k_label = self.norm_data(w2k_label, data_type='w2k')
        return w2k_x, w2k_y, inp_pred_surface_data, w2k_label
    
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
                #and (self.glob_step-1) % self.log_step == 0
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
            if self.sample_mode == 'FAST':
                margin_x,margin_y, margin_data = self.get_margin_grid(era5_data_surface_input, h8_data)
                margin_x = margin_x.to(self.device)
                margin_y = margin_y.to(self.device)
                margin_data = margin_data.to(self.device)
            elif self.sample_mode == 'SLOW':
                margin_x = data['margin_x'].to(self.device)
                margin_y = data['margin_y'].to(self.device)
                margin_data = data['inp_data_surface'].to(self.device)
            
            # grid supervision
            if self.decoder_type == 'MULTI_VAR':
                margin_input = self.encoding_coord(margin_x, margin_y)
            else:
                margin_input = self.PoseEncoding(margin_x, margin_y, 8)
            pred = self.model.forward(h8_data, era5_data_surface_input, margin_input, margin_data, 'grid', self.device)
            if self.decoder_type == 'MULTI_VAR':
                pred = pred.permute(0,2,1).view(self.config.train_cfg.batch_size,5,self.sample_num,self.out_lon_size,self.out_lat_size)
            # cal mean value for each pixel and add shortcut
            pred_mean_s = torch.mean(pred, axis=2)
            H, W = pred.shape[-2:]
            era5_data_surface_interp = nn.functional.interpolate(era5_data_surface_input, align_corners=True, size=[H, W], mode='bilinear')
            pred_mean_s = pred_mean_s + era5_data_surface_interp
            
            loss_baseline = self.grid_criterion(era5_data_surface_interp, era5_data_surface_label[:,:,:-1,:-1])
            train_loss_dict['baseline_grid_loss'] += loss_baseline
            
            if self.with_gt:
                loss_pred_gt = self.grid_criterion(pred_mean_s, era5_data_surface_label[:,:,:-1,:-1])
                train_loss_dict['pred_grid_loss'] += loss_pred_gt
                train_grid_loss = loss_pred_gt
            else:
                H, W = era5_data_surface_input.shape[-2:]
                pred_mean = F.adaptive_avg_pool2d(pred_mean_s, [H-1, W-1])
                loss_surface = self.grid_criterion(pred_mean, era5_data_surface_input[:,:,:-1,:-1])
                loss_pred_interp = self.grid_criterion(pred_mean_s, era5_data_surface_interp)                   
                train_grid_loss = 0.1*loss_pred_interp + 0.9*loss_surface
                train_loss_dict['pred_grid_loss'] += train_grid_loss 
            
            w2k_x = data['w2k_x'].to(self.device)
            w2k_y = data['w2k_y'].to(self.device)
            w2k_data = data['w2k_interp'].to(self.device)
            w2k_label = data['w2k_label'].to(self.device)
            if self.decoder_type == 'MULTI_VAR':
                w2k_input = self.encoding_coord(w2k_x, w2k_y)
            else:
                w2k_input = self.PoseEncoding(w2k_x, w2k_y, 8)
            pred = self.model.forward(h8_data, era5_data_surface_input, w2k_input, w2k_data, 'stn', self.device, w2k_x=w2k_x, w2k_y=w2k_y, dx=self.dx, dy=self.dy)
            if self.decoder_type == 'MULTI_VAR':
                pred = pred.permute(0,2,1)
            pred = pred + w2k_data.permute(0,2,1).contiguous()
            pred_wind = torch.sqrt(pred[:,0,:]**2 + pred[:,1,:]**2)
            
            w2k_interp_ot = w2k_data.permute(0,2,1)[:,2:,:].contiguous()
            w2k_interp_wind = torch.sqrt(w2k_data.permute(0,2,1)[:,0,:].contiguous()**2 + w2k_data.permute(0,2,1)[:,1,:].contiguous()**2)
            
            w2k_loss_ot = self.stn_criterion(w2k_label.permute(0,2,1)[:,1:,:].contiguous(), pred[:,2:,:])
            w2k_loss_wind = self.stn_criterion(w2k_label[:,:,0], pred_wind)
            
            w2k_interp_loss_ot = self.stn_criterion(w2k_label.permute(0,2,1)[:,1:,:].contiguous(), w2k_interp_ot)
            w2k_interp_loss_wind = self.stn_criterion(w2k_label.permute(0,2,1)[:,0,:].contiguous(), w2k_interp_wind)

            train_stn_loss = w2k_loss_ot + w2k_loss_wind
            train_loss_dict['pred_station_loss'] += train_stn_loss
            interp_stn_loss = w2k_interp_loss_ot + w2k_interp_loss_wind
            train_loss_dict['baseline_station_loss'] += interp_stn_loss
            if self.accelerator.is_main_process:
                self.log_writer.add_scalars(main_tag='training/train_gird_loss', 
                                            tag_scalar_dict={'loss_pred':train_grid_loss.item(),
                                                            'loss_interp_baseline':loss_baseline.item()}, 
                                            global_step=self.glob_step)
                  
                self.log_writer.add_scalars(main_tag='training/train_stn_loss',
                                            tag_scalar_dict={'loss_pred':train_stn_loss.item(),
                                                            'loss_interp_baseline':interp_stn_loss.item()},
                                            global_step= self.glob_step)
                
            if self.with_stn:
                train_loss = train_grid_loss + 0.05*train_stn_loss
            else:
                train_loss = train_grid_loss + 0.*train_stn_loss
                # train_stn_loss = 0
            
            self.optimizer.zero_grad()
            self.accelerator.backward(train_loss)
            self.optimizer.step()
            torch.cuda.empty_cache()

            if (self.glob_step-1) % (self.log_step*10) == 0:

                # self.accelerator.print(f"[Epoch:{epoch}/{self.num_epoch}][batch:{i}/{len(self.train_dataloader)}]: loss_baseline:{loss_baseline}/gird_loss:{loss_pred_gt}/stn_interp_loss:{interp_stn_loss}/train_stn_loss:{train_stn_loss}")

                if self.accelerator.is_main_process:
                    if self.with_vis:
                        pred_inver = self.inverse_norm(pred_mean_s)
                        era5_input_inver = self.inverse_norm(era5_data_surface_input)
                        era5_gt_inver = self.inverse_norm(era5_data_surface_label[:,:,:-1,:-1])
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
                era5_data_surface_input = data['era5_data_surface_input'][:,0,:,:,:].to(self.device)
                era5_data_surface_label = data['era5_data_surface_label'][:,0,:,:,:].to(self.device)
                if not self.with_h8:
                    h8_data = torch.zeros(h8_data.shape).to(self.device)
                
                if self.sample_mode == 'FAST':
                    margin_x,margin_y, margin_data = self.get_margin_grid(era5_data_surface_input, h8_data)
                    margin_x = margin_x.to(self.device)
                    margin_y = margin_y.to(self.device)
                    margin_data = margin_data.to(self.device)
                elif self.sample_mode == 'SLOW':
                    margin_x = data['margin_x'].to(self.device)
                    margin_y = data['margin_y'].to(self.device)
                    margin_data = data['inp_data_surface'].to(self.device)
                if self.decoder_type == 'MULTI_VAR':
                    margin_input = self.encoding_coord(margin_x, margin_y)
                else:
                    margin_input = self.PoseEncoding(margin_x, margin_y, 8)
                pred = self.model.forward(h8_data, era5_data_surface_input, margin_input, margin_data, 'grid', self.device)
                if self.decoder_type == 'MULTI_VAR':
                    pred = pred.permute(0,2,1).view(self.config.train_cfg.batch_size,5,self.sample_num,self.out_lon_size,self.out_lat_size)
            

                pred_mean_s = torch.mean(pred, axis=2)
                H, W = pred.shape[-2:]
                era5_data_surface_interp = nn.functional.interpolate(era5_data_surface_input, align_corners=True, size=[H, W], mode='bilinear')
                pred_mean_s = pred_mean_s + era5_data_surface_interp

                loss_baseline = self.grid_criterion(era5_data_surface_interp, era5_data_surface_label[:,:,:-1,:-1])
                valid_loss_dict['baseline_grid_loss'] += loss_baseline


                if self.with_gt:
                    loss_pred_gt = self.grid_criterion(pred_mean_s, era5_data_surface_label[:,:,:-1,:-1])
                    valid_loss_dict['pred_grid_loss'] += loss_pred_gt
                    valid_grid_loss = loss_pred_gt
                else:
                    H, W = era5_data_surface_input.shape[-2:]
                    pred_mean = F.adaptive_avg_pool2d(pred_mean_s, [H-1, W-1])
                    loss_surface = self.grid_criterion(pred_mean, era5_data_surface_input[:,:,:-1,:-1])
                    loss_pred_interp = self.grid_criterion(pred_mean_s, era5_data_surface_interp)
                    valid_loss_dict['pred_grid_loss'] += 0.1*loss_pred_interp + 0.9*loss_surface
                
                # station supervision
                w2k_x = data['w2k_x'].to(self.device)
                w2k_y = data['w2k_y'].to(self.device)
                w2k_data = data['w2k_interp'].to(self.device)
                w2k_label = data['w2k_label'].to(self.device)
                if self.decoder_type == 'MULTI_VAR':
                    w2k_input = self.encoding_coord(w2k_x, w2k_y)
                else:
                    w2k_input = self.PoseEncoding(w2k_x, w2k_y, 8)
                pred = self.model.forward(h8_data, era5_data_surface_input, w2k_input, w2k_data, 'stn', self.device, w2k_x=w2k_x, w2k_y=w2k_y, dx=self.dx, dy=self.dy)
                if self.decoder_type == 'MULTI_VAR':
                    pred = pred.permute(0,2,1)
                pred = pred + w2k_data.permute(0,2,1).contiguous()
                pred_wind = torch.sqrt(pred[:,0,:]**2 + pred[:,1,:]**2)
                
                w2k_interp_ot = w2k_data.permute(0,2,1)[:,2:,:].contiguous()
                w2k_interp_wind = torch.sqrt(w2k_data.permute(0,2,1)[:,0,:].contiguous()**2 + w2k_data.permute(0,2,1)[:,1,:].contiguous()**2)
                
                w2k_loss_ot = self.stn_criterion(w2k_label.permute(0,2,1)[:,1:,:].contiguous(), pred[:,2:,:])
                w2k_loss_wind = self.stn_criterion(w2k_label[:,:,0], pred_wind)
                
                w2k_interp_loss_ot = self.stn_criterion(w2k_label.permute(0,2,1)[:,1:,:].contiguous(), w2k_interp_ot)
                w2k_interp_loss_wind = self.stn_criterion(w2k_label.permute(0,2,1)[:,0,:].contiguous(), w2k_interp_wind)

                valid_stn_loss = w2k_loss_ot + w2k_loss_wind
                valid_loss_dict['pred_station_loss'] += valid_stn_loss
                interp_stn_loss = w2k_interp_loss_ot + w2k_interp_loss_wind
                valid_loss_dict['baseline_station_loss'] += interp_stn_loss

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
            self.accelerator.print(f"=>[Epoch:{epoch}/{self.num_epoch}]")
            self.accelerator.print(f"==>[train_loss]:\nbaseline_pred_loss:{train_loss_dict['baseline_grid_loss']/len(self.train_dataloader)}\npred_grid_loss:{train_loss_dict['pred_grid_loss']/len(self.train_dataloader)}\nbaseline_station_loss:{train_loss_dict['baseline_station_loss']/len(self.train_dataloader)}\npred_station_loss:{train_loss_dict['pred_station_loss']/len(self.train_dataloader)}")
            # self.train_one_epoch(epoch)
            self.model.eval()
            val_loss_dict = self.valid_one_epoch(epoch)
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
                
                margin_x,margin_y, margin_data = self.get_margin_grid(era5_data_surface_input, h8_data)
            
                margin_x = margin_x.to(self.device)
                margin_y = margin_y.to(self.device)
                margin_data = margin_data.to(self.device)
                
                # grid supervision
                if self.decoder_type == 'MULTI_VAR':
                    margin_input = self.encoding_coord(margin_x, margin_y)
                else:
                    margin_input = self.PoseEncoding(margin_x, margin_y, 8)
                pred = self.model.forward(h8_data, era5_data_surface_input, margin_input, margin_data, 'grid', self.device)
                if self.decoder_type == 'MULTI_VAR':
                    pred = pred.permute(0,2,1).view(self.config.train_cfg.batch_size,5,self.sample_num,self.out_lon_size,self.out_lat_size)
            
                # cal mean value for each pixel and add shortcut
                pred_mean_s = torch.mean(pred, axis=2)
                H, W = pred.shape[-2:]
                era5_data_surface_interp = nn.functional.interpolate(era5_data_surface_input, align_corners=True, size=[H, W], mode='bilinear')
                pred_mean_s = pred_mean_s + era5_data_surface_interp
                
                loss_baseline = self.grid_criterion(era5_data_surface_interp, era5_data_surface_label[:,:,:-1,:-1])
                test_loss_dict['baseline_grid_loss'] += loss_baseline


                if self.with_gt:
                    loss_pred_gt = self.grid_criterion(pred_mean_s, era5_data_surface_label[:,:,:-1,:-1])
                    test_loss_dict['pred_grid_loss'] += loss_pred_gt
                else:
                    H, W = era5_data_surface_input.shape[-2:]
                    pred_mean = F.adaptive_avg_pool2d(pred_mean_s, [H-1, W-1])
                    loss_surface = self.grid_criterion(pred_mean, era5_data_surface_input[:,:,:-1,:-1])
                    test_loss_dict['pred_grid_mean_loss'] += loss_surface

                    loss_pred_interp = self.grid_criterion(pred_mean_s, era5_data_surface_interp)
                    test_loss_dict['pred_grid_interp_loss'] += loss_pred_interp
                    
                # pred_inver = self.inverse_norm(pred_mean_s)
                pred_inver = self.inverse_norm(pred_mean_s)
                
                
                w2k_x = data['w2k_x'].to(self.device)
                w2k_y = data['w2k_y'].to(self.device)
                w2k_data = data['w2k_interp'].to(self.device)
                w2k_label = data['w2k_label'].to(self.device)
                if self.decoder_type == 'MULTI_BLOCK':
                    w2k_input = self.PoseEncoding(w2k_x, w2k_y, 8)
                else:
                    w2k_input = self.encoding_coord(w2k_x, w2k_y)
                pred = self.model.forward(h8_data, era5_data_surface_input, w2k_input, w2k_data, 'stn', self.device, w2k_x=w2k_x, w2k_y=w2k_y, dx=self.dx, dy=self.dy)
                if self.decoder_type == 'MULTI_VAR':
                    pred = pred.permute(0,2,1)
                pred.to(self.device)
                pred = pred + w2k_data.permute(0,2,1).contiguous()
                pred_wind = torch.sqrt(pred[:,0,:]**2 + pred[:,1,:]**2)
                
                w2k_interp_ot = w2k_data.permute(0,2,1)[:,2:,:].contiguous()
                w2k_interp_wind = torch.sqrt(w2k_data.permute(0,2,1)[:,0,:].contiguous()**2 + w2k_data.permute(0,2,1)[:,1,:].contiguous()**2)
                
                #w2k_loss_ot = self.stn_criterion(w2k_label.permute(0,2,1)[:,1:,:].contiguous(), pred[:,2:,:])
                
                tmp_label = self.inverse_norm(w2k_label[:,:,0],data_type='w2k',var_name='wind')
                tmp_pred = self.inverse_norm(pred_wind,data_type='w2k',var_name='wind')
                w2k_mse_wind = F.mse_loss(torch.from_numpy(tmp_label),torch.from_numpy(tmp_pred),reduction='mean')
                w2k_mae_wind = F.l1_loss(torch.from_numpy(tmp_label),torch.from_numpy(tmp_pred),reduction='mean')
                w2k_matric_dict['mse']['wind'] += w2k_mse_wind
                w2k_matric_dict['mae']['wind'] += w2k_mae_wind
                tmp_label = self.inverse_norm(w2k_label.permute(0,2,1)[:,1,:].contiguous(),data_type='w2k',var_name='t2m')
                tmp_pred = self.inverse_norm(pred[:,2,:],data_type='w2k',var_name='t2m')
                w2k_mse_t2m = F.mse_loss(torch.from_numpy(tmp_label),torch.from_numpy(tmp_pred),reduction='mean')
                w2k_mae_t2m = F.l1_loss(torch.from_numpy(tmp_label),torch.from_numpy(tmp_pred),reduction='mean')
                w2k_matric_dict['mse']['t2m'] += w2k_mse_t2m
                w2k_matric_dict['mae']['t2m'] += w2k_mae_t2m
                tmp_label = self.inverse_norm(w2k_label.permute(0,2,1)[:,2,:].contiguous(),data_type='w2k',var_name='sp')
                tmp_pred = self.inverse_norm(pred[:,3,:],data_type='w2k',var_name='sp')
                w2k_mse_sp = F.mse_loss(torch.from_numpy(tmp_label)/100,torch.from_numpy(tmp_pred)/100,reduction='mean')
                w2k_mae_sp = F.l1_loss(torch.from_numpy(tmp_label)/100,torch.from_numpy(tmp_pred)/100,reduction='mean')
                w2k_matric_dict['mse']['sp'] += w2k_mse_sp
                w2k_matric_dict['mae']['sp'] += w2k_mae_sp
                tmp_label = self.inverse_norm(w2k_label.permute(0,2,1)[:,3,:].contiguous(),data_type='w2k',var_name='tp1h')
                tmp_pred = self.inverse_norm(pred[:,4,:],data_type='w2k',var_name='tp1h')
                
                w2k_mse_tp1h = F.mse_loss(torch.from_numpy(tmp_label),torch.from_numpy(tmp_pred),reduction='mean')
                w2k_mae_tp1h = F.l1_loss(torch.from_numpy(tmp_label),torch.from_numpy(tmp_pred),reduction='mean')
                w2k_matric_dict['mse']['tp1h'] += w2k_mse_tp1h
                w2k_matric_dict['mae']['tp1h'] += w2k_mae_tp1h
                
                if i == 200:
                    vis_stn_loss_dict = {
                        'wind': [],
                        't2m': [],
                        'sp': [],
                        'tp1h': []
                        }
                    for stn_id in range(pred.shape[2]):
                        wind_loss_tmp = F.mse_loss(pred_wind[:,stn_id],w2k_label[:,stn_id,0])
                        vis_stn_loss_dict['wind'].append(wind_loss_tmp.item())
                        
                        t2m_loss_tmp = F.mse_loss(pred[:,2,stn_id], w2k_label[:,stn_id,1])
                        vis_stn_loss_dict['t2m'].append(t2m_loss_tmp.item())
                        
                        sp_loss_tmp = F.mse_loss(pred[:,3,stn_id], w2k_label[:,stn_id,2])
                        vis_stn_loss_dict['sp'].append(sp_loss_tmp.item())

                        tp1h_loss_tmp = F.mse_loss(pred[:,4,stn_id], w2k_label[:,stn_id,3])
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
            
            self.accelerator.print(f"==>[test_loss]:{w2k_matric_dict},data_len:{dataset_len}")
            import pdb
            pdb.set_trace()          
                        
                    

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
        
    def norm_data(self, data, data_type='h8', norm_type='mean_std'):
        
        if norm_type.lower() == 'mean_std':
            if data_type.lower() == 'h8':
                norm_dict = np.load(self.h8_norm_data_path, allow_pickle=True).item()
                mean = norm_dict['mean']
                mean = np.expand_dims(mean, axis=[0,2,3])
                std = norm_dict['std']
                std = np.expand_dims(std, axis=[0,2,3])
                data = (data - mean) / std
            elif data_type.lower() == 'pred':
                
                with open(self.pred_single_norm_path, mode='r') as f:
                    single_level_mean_std = json.load(f)
                with open(self.pred_pressure_norm_path, mode='r') as f:
                    pressure_level_mean_std = json.load(f)
                

                if data.shape[1] == 5:
                    for idx, var_name in enumerate(self.train_cfg.pred_names.surface):
                        # [np.newaxis,:,np.newaxis,np.newaxis]
                        mean = np.array(single_level_mean_std['mean'][var_name])
                        std = np.array(single_level_mean_std['std'][var_name])
                        data[:,idx,:,:] = (data[:,idx,:,:] - mean) / std
                else:
                    for idx, var_name in enumerate(self.train_cfg.pred_names.pressure):
                        name_list = var_name.split('-')
                        level_index = self.height_level_list.index(float(name_list[1]))
                        level_index = self.height_level_indexes[level_index]
                        
                        mean = np.array([pressure_level_mean_std['mean'][name_list[0]][level_index]])[np.newaxis,:,np.newaxis,np.newaxis]
                        std = np.array([pressure_level_mean_std['std'][name_list[0]][level_index]])[np.newaxis,:,np.newaxis,np.newaxis]
                        
                        data[:,idx,:,:] = (data[:,idx,:,:] - mean) / std
            elif data_type.lower() == 'w2k':
                
                with open(self.config.train_cfg.norm_path.pred_single_norm_path, mode='r') as f:
                    single_level_mean_std = json.load(f)
                    
                mean_wind = np.sqrt(single_level_mean_std['mean']['u10']**2+single_level_mean_std['mean']['v10']**2)
                std_wind = np.sqrt(single_level_mean_std['std']['u10']**2+single_level_mean_std['std']['v10']**2)
                data[:,0] = (data[:,0] - mean_wind) / std_wind

                mean_t2 = single_level_mean_std['mean']['t2m']
                std_t2 = single_level_mean_std['std']['t2m']
                data[:,1] = (data[:,1] - mean_t2) / std_t2

                mean_sp = single_level_mean_std['mean']['sp']
                std_sp = single_level_mean_std['std']['sp']
                data[:,2] = (data[:,2] - mean_sp) / std_sp

                mean_tp1h = single_level_mean_std['mean']['tp1h']
                std_tp1h = single_level_mean_std['std']['tp1h']
                data[:,3] = (data[:,3] - mean_tp1h) / std_tp1h

        else:
            raise NotImplementedError

        return data
    
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
                
        
        