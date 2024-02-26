import pandas as pd
import os
import numpy as np
import torch
import torch.nn.functional as F
import yaml
import xarray as xr
from easydict import EasyDict
import random
import io, pickle
import sys
import json
sys.path.append('/mnt/petrelfs/liuzili/code/OBDS')
import utils.mypath as mypath
from petrel_client.client import Client
import math
from multiprocessing import shared_memory,Pool
import multiprocessing as mp
import copy
import queue,time

class DARS_INF_Downscale_Dataset(torch.utils.data.Dataset):
    def __init__(self, train_config, type):
        super().__init__()
        self.client = Client(conf_path="/mnt/petrelfs/liuzili/petreloss.conf")
        self.train_cfg = train_config
        self.pred_data_path = train_config.pred_data_path
        self.sample_mode = train_config.sample_mode
        self.h8_data_path = train_config.h8_data_path
        self.era5_data_path = train_config.era5_data_path
        self.w2k_data_path = train_config.w2k_data_path
        self.h8_norm_data_path = train_config.norm_path.h8_norm_data_path   # dict{'mean', 'std'}
        self.pred_single_norm_path = train_config.norm_path.pred_single_norm_path
        self.pred_names = self.train_cfg.pred_names
        self.dx = train_config.dx
        self.dy = train_config.dy
        self.downscale_type = self.train_cfg.downscale_type

        self.type = type
        if self.type == 'train':
            w2k_file = os.path.join(self.w2k_data_path, 'w2k_train.npy')
            self.w2k_data = np.load(w2k_file)
            time_span = self.train_cfg.train_time_span
        elif self.type == 'valid':
            w2k_file = os.path.join(self.w2k_data_path, 'w2k_valid.npy')
            self.w2k_data = np.load(w2k_file)
            time_span = self.train_cfg.valid_time_span
        elif self.type == 'test':
            w2k_file = os.path.join(self.w2k_data_path, 'w2k_test.npy')
            self.w2k_data = np.load(w2k_file)
            time_span = self.train_cfg.test_time_span
        else:
            raise NotImplementedError

        self.start_time = time_span[0]
        self.end_time = time_span[1]
        self.time_list = pd.date_range(self.start_time, self.end_time, freq=str(self.train_cfg.pred_window_len)+'H')
        
        # print(f"loading {type} data files......")
        if self.train_cfg.data_file_arxiv_flag is True:
            if type == 'train':
                file_dict = np.load(self.train_cfg.data_file_arxiv.train_files, allow_pickle=True).item()
                self.h8_files = file_dict['h8_files']
                self.h8_delta_t = file_dict['h8_delta_t']
                self.pred_files = file_dict['pred_files']
                self.era5_files = file_dict['era5_files']
            elif type == 'valid':
                file_dict = np.load(self.train_cfg.data_file_arxiv.valid_files, allow_pickle=True).item()
                self.h8_files = file_dict['h8_files']
                self.h8_delta_t = file_dict['h8_delta_t']
                self.pred_files = file_dict['pred_files']
                self.era5_files = file_dict['era5_files']
            elif type == 'test':
                file_dict = np.load(self.train_cfg.data_file_arxiv.test_files, allow_pickle=True).item()
                self.h8_files = file_dict['h8_files'][:-8]
                self.h8_delta_t = file_dict['h8_delta_t'][:-8]
                self.pred_files = file_dict['pred_files'][:-8]
                self.era5_files = file_dict['era5_files'][:-8]
            
        else:
            self.h8_files, self.h8_delta_t, self.pred_files, self.era5_files = self.get_input_files()
            file_dict = {}
            file_dict['h8_files'] = self.h8_files
            file_dict['h8_delta_t'] = self.h8_delta_t
            file_dict['pred_files'] = self.pred_files
            file_dict['era5_files'] = self.era5_files
            
            np.save(f'/mnt/petrelfs/liuzili/code/DA_RS/arxiv_file_list/split_time/{type}_files.npy', file_dict)
        
        
        self.pred_ahead = self.train_cfg.forecast_ahead_list[0]

        self.h8_size = self.train_cfg.h8_size
        self.pred_size  = self.train_cfg.pred_size
        self.era5_size = self.train_cfg.era5_size

        self.h8_downsample_scale = self.train_cfg.h8_downsample_scale
        self.pred_downsample_scale = self.train_cfg.pred_downsample_scale

        self.lon_range = train_config.lon_range
        self.lat_range = train_config.lat_range
        self.input_resolution = train_config.input_resolution
        self.target_resolution = train_config.target_resolution
        
        self.has_normed = False

        #===========init multi processing===========
        # print("loading dataset from mp")
        self.data_element_num = 8
        self.index_queue = mp.Queue()
        self.unit_data_queue = mp.Queue()

        self.index_queue.cancel_join_thread()
        self.unit_data_queue.cancel_join_thread()

        self.compound_data_queue = []
        self.sharedmemory_list = []
        self.compound_data_queue_dict = {}
        self.sharedmemory_dict = {}

        self.compound_data_queue_num = 8

        self.lock = mp.Lock()
        
        self.h8 = np.zeros((self.data_element_num, self.h8_size[0], self.h8_size[1]), dtype=np.float32)

        for _ in range(self.compound_data_queue_num):
            self.compound_data_queue.append(mp.Queue())
            shm = shared_memory.SharedMemory(create=True, size=self.h8.nbytes)
            self.sharedmemory_list.append(shm)
        
        self.arr = mp.Array('i', range(self.compound_data_queue_num))

        self._workers = []

        for _ in range(8):
            w = mp.Process(target=self.load_data_process)
            w.daemon = True
            w.start()
            self._workers.append(w)
        w = mp.Process(target=self.data_compound_process)
        w.daemon = True
        w.start()
        self._workers.append(w)
        
        #result_dict = self.__getitem__(1)
        
        # print("FIN")
        
    def load_data_process(self):
        while True:
            job_pid, file, frame_id = self.index_queue.get()
            if job_pid not in self.compound_data_queue_dict:
                try:
                    self.lock.acquire()
                    for i in range(self.compound_data_queue_num):
                        if job_pid == self.arr[i]:
                            self.compound_data_queue_dict[job_pid] = self.compound_data_queue[i]
                            self.sharedmemory_dict[job_pid] = self.sharedmemory_list[i]
                            break
                    if (i == self.compound_data_queue_num - 1) and job_pid != self.arr[i]:
                        print("error", job_pid, self.arr)
                except Exception as err:
                    raise err
                finally:
                    self.lock.release()
            
            b = np.ndarray(self.h8.shape, dtype=self.h8.dtype, buffer=self.sharedmemory_dict[job_pid].buf)
            # start_time = time.time()
            with io.BytesIO(self.client.get(file)) as f:
                unit_data = np.load(f)
            # end_time = time.time()
            # print("h8_laoding time:", end_time-start_time)
            b[frame_id] = unit_data
            self.unit_data_queue.put((job_pid, file, frame_id))

    def data_compound_process(self):
        recorder_dict = {}
        while True:
            job_pid, file, frame_id = self.unit_data_queue.get()
            if job_pid not in self.compound_data_queue_dict:
                try:
                    self.lock.acquire()
                    for i in range(self.compound_data_queue_num):
                        if job_pid == self.arr[i]:
                            self.compound_data_queue_dict[job_pid] = self.compound_data_queue[i]
                            break
                    if (i == self.compound_data_queue_num - 1) and job_pid != self.arr[i]:
                        print("error", job_pid, self.arr)
                except Exception as err:
                    raise err
                finally:
                    self.lock.release()
            if (job_pid) in recorder_dict:
                recorder_dict[(job_pid)][(file, frame_id)] = 1
            else:
                recorder_dict[(job_pid)] = {(file, frame_id): 1}
            # print("recorder_len", len(recorder_dict[job_pid]))
            # print("recorder_dict", recorder_dict)
            if len(recorder_dict[job_pid]) == self.data_element_num:
                del recorder_dict[job_pid]
                self.compound_data_queue_dict[job_pid].put((file))

    def downsample(self, data, scale, type='direct'):
        if type == 'direct':
            data = data[:,:,::scale,::scale].contiguous()
        elif type == 'avgpool':
            out_h = math.ceil(data.shape[2] / scale)
            out_w = math.ceil(data.shape[3] / scale)
            data = F.interpolate(data, size=(out_h, out_w), mode='bilinear').contiguous()
        else:
            raise NotImplementedError
        return data

    def get_data(self, files, type):
        
        job_pid = os.getpid()
        
        if job_pid not in self.compound_data_queue_dict:
            try:
                self.lock.acquire()
                for i in range(self.compound_data_queue_num):
                    if i == self.arr[i]:
                        self.arr[i] = job_pid
                        self.compound_data_queue_dict[job_pid] = self.compound_data_queue[i]
                        self.sharedmemory_dict[job_pid] = self.sharedmemory_list[i]
                        break
                if (i == self.compound_data_queue_num - 1) and job_pid != self.arr[i]:
                    print("error", job_pid, self.arr)

            except Exception as err:
                raise err
            finally:
                self.lock.release()
        try:
            file = self.compound_data_queue_dict[job_pid].get(False)
            raise ValueError
        except queue.Empty:
            pass
        except Exception as err:
            raise err
        if type == 'h8':
            b = np.ndarray(self.h8.shape, dtype=self.h8.dtype, buffer=self.sharedmemory_dict[job_pid].buf)
            for frame_id, file in enumerate(files):
                self.index_queue.put((job_pid, file, frame_id))
            file = self.compound_data_queue_dict[job_pid].get()
            data_tmp = copy.deepcopy(b)
            data_tmp[np.isnan(data_tmp)] = 0
            if (not self.has_normed):
                data_tmp = self.norm_data(data_tmp.reshape(2,4,720,1120), data_type='h8')
            
            data_tmp = torch.from_numpy(data_tmp).float().contiguous()

        elif type == 'era5':
            pass
        else:
            raise NotImplementedError
        return data_tmp
    
    def __getitem__(self, item):
        h8_idx = item % len(self.h8_files)
        h8_file = self.h8_files[h8_idx]
        h8_delta_t = self.h8_delta_t[h8_idx] # list

        era5_idx = item % len(self.era5_files)
        era5_file = self.era5_files[era5_idx]

        time_str = era5_file[0][0].split('/')[-2]+' '+era5_file[0][0].split('/')[-1][:8] 
        current_time = pd.to_datetime(time_str) + pd.Timedelta(hours=8)
        ori_time = pd.to_datetime('2017-01-01 00:00:00')
        time_idx = (current_time - ori_time).total_seconds()/60/60
        
        w2k_item = self.w2k_data[:, :, int(time_idx)]
        w2k_item = w2k_item[w2k_item[:,1]>=self.lon_range[0]]
        h8_file = [file.replace('cluster1','cluster2') for file in h8_file]
        
        h8_data = self.get_data(h8_file, 'h8')
        #h8_data = self.get_h8_data(h8_file) 
        h8_data_lr = self.downsample(h8_data, self.h8_downsample_scale, type=self.downscale_type) 
        h8_data_lr = h8_data[:,:,::self.h8_downsample_scale,::self.h8_downsample_scale].contiguous()
        
        f,c,h,w = h8_data_lr.shape
        tmp = torch.zeros(f,c,h+1,w+1)
        tmp[:,:,:h,:w] = h8_data_lr
        tmp[:,:,-1,-1] = h8_data[:,:,-1,-1]
        h8_data = tmp
        del tmp
        del h8_data_lr
        tmp = torch.zeros(h8_data.shape)
        h8_data = tmp
        del tmp
        
        era5_data_surface_label, _ = self.get_era5_data(era5_file) # (da_window, var_num, 481, 481)
        
        era5_data_surface_input = self.downsample(era5_data_surface_label, self.pred_downsample_scale, type=self.downscale_type)
        
        if self.sample_mode == 'SLOW':
            margin_x, margin_y, inp_data_surface = self.get_margin_gird(era5_data_surface_input, h8_data)
        else:
            margin_x, margin_y, inp_data_surface = torch.zeros(0), torch.zeros(0), torch.zeros(0)
        
        w2k_x, w2k_y, w2k_interp, w2k_label = self.get_stn_data(era5_data_surface_input, w2k_item)
        return {
            'h8_data': h8_data, 'h8_delta_t': h8_delta_t, 
            'margin_x': margin_x, 'margin_y': margin_y, 'inp_data_surface': inp_data_surface,
            'w2k_x': w2k_x, 'w2k_y': w2k_y, 'w2k_interp': w2k_interp, 'w2k_label': w2k_label,
            'era5_data_surface_input': era5_data_surface_input,'era5_data_surface_label': era5_data_surface_label
            }
          
    def get_stn_data(self, pred_data_surface, w2k_item):
        _, var_name, LH, LW = pred_data_surface.shape
        in_lon = self.lon_range[0] + np.array(range(LW)) * self.input_resolution
        in_lat = self.lat_range[0] + np.array(range(LH)) * self.input_resolution
        in_lat = in_lat[::-1]
        p_lon = w2k_item[:,1]
        p_lat = w2k_item[:,0]
        px = (p_lon-self.lon_range[0])/self.target_resolution
        py = (p_lat-self.lat_range[0])/self.target_resolution
        
        inp_pred_surface_data = []
        for channel in range(pred_data_surface.shape[1]):
            # y_len, x_len = pred_data_surface[0,channel,:,:].shape
            # assert len(in_lon) == x_len and len(in_lat) == y_len
            coord_x = in_lon
            coord_y = in_lat
            data = xr.DataArray(data=pred_data_surface[0,channel,:,:], dims=['y','x'], coords=(coord_y.tolist(), coord_x.tolist()))
            var_list = data.interp(x=xr.DataArray(p_lon, dims='z'),
                                   y=xr.DataArray(p_lat, dims='z'))
            inp_pred_surface_data.append(var_list)
        w2k_label = []
        
        w2k_label.append(w2k_item[:, 16])
        w2k_label.append(w2k_item[:, 4]+273.15)
        w2k_label.append(w2k_item[:, 3]*100)
        w2k_label.append(w2k_item[:, 10])
        w2k_label = np.stack(w2k_label, axis=-1)
        
        inp_pred_surface_data = np.stack(inp_pred_surface_data, axis=-1)
        w2k_x = np.array(px) * self.dx
        w2k_y = np.array(py) * self.dy
        
        inp_pred_surface_data = torch.from_numpy(inp_pred_surface_data).float()
        w2k_label = torch.from_numpy(w2k_label).float()
        w2k_x = torch.from_numpy(w2k_x).float()
        w2k_y = torch.from_numpy(w2k_y).float()
        if (not self.has_normed):
            w2k_label = self.norm_data(w2k_label, data_type='w2k')
        
        return w2k_x, w2k_y, inp_pred_surface_data, w2k_label
    
    def get_margin_gird(self, pred_data_surface, h8_data):
        _, _, H, W = h8_data.shape
        _, _, LH, LW = pred_data_surface.shape
        
        px, py = np.meshgrid(np.arange(W-1),np.arange(H-1))
        num_samples = 10
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
        # import pdb
        # pdb.set_trace()
        inp_pred_surface_data = []
        for channel in range(pred_data_surface.shape[1]):
            y_len, x_len = pred_data_surface[0,channel,:,:].shape
            # assert len(in_lon) == x_len and len(in_lat) == y_len
            coord_x = in_lon
            coord_y = in_lat
            data = xr.DataArray(data=pred_data_surface[0,channel,:,:], dims=['y','x'], coords=(coord_y.tolist(), coord_x.tolist()))
            var_list = data.interp(x=xr.DataArray(p_lon, dims='z'),
                                   y=xr.DataArray(p_lat, dims='z'))
            inp_pred_surface_data.append(var_list)
        inp_pred_surface_data = np.stack(inp_pred_surface_data, axis=-1)
        margin_x = np.array(px) * self.dx
        margin_y = np.array(py) * self.dy
        del px,py,p_lon,p_lat,in_lon,in_lat
        inp_pred_surface_data = torch.from_numpy(inp_pred_surface_data).float()
        margin_x = torch.from_numpy(margin_x).float()
        margin_y = torch.from_numpy(margin_y).float()
        '''
        margin_x_list = []
        margin_y_list = []
        for x in range(0, self.output_lon_size):
            for y in range(0, self.output_lat_size):
                margin_x_list.append(x)
                margin_y_list.append(y)

        margin_lon = self.begin_lon + np.array(margin_x_list) * self.resolution
        margin_lat = self.begin_lat + np.array(margin_y_list) * self.resolution

        inp_data_surface = []
        inp_data_pressure = []
        for channel in range(pred_data_surface.shape[1]):
            y_len, x_len = pred_data_surface[0,channel,:,:].shape
            assert len(self.pred_lon) == x_len and len(self.pred_lat) == y_len
            coord_x = self.pred_lon
            coord_y = self.pred_lat
            data = xr.DataArray(data=pred_data_surface[0,channel,:,:], dims=['y','x'], coords=(coord_y.tolist(), coord_x.tolist()))
            var_list = data.interp(x=xr.DataArray(margin_lon, dims='z'),
                                   y=xr.DataArray(margin_lat, dims='z'))
            inp_data_surface.append(var_list)
        for channel in range(pred_data_pressure.shape[1]):
            y_len, x_len = pred_data_pressure[0,channel,:,:].shape
            assert len(self.pred_lon) == x_len and len(self.pred_lat) == y_len
            coord_x = self.pred_lon
            coord_y = self.pred_lat
            data = xr.DataArray(data=pred_data_pressure[0,channel,:,:], dims=['y','x'], coords=(coord_y.tolist(), coord_x.tolist()))
            var_list = data.interp(x=xr.DataArray(margin_lon, dims='z'),
                                   y=xr.DataArray(margin_lat, dims='z'))
            inp_data_pressure.append(var_list)
        
        
        inp_data_pressure = torch.from_numpy(inp_data_pressure).float()
        
        '''
        return margin_x, margin_y, inp_pred_surface_data
        
    def get_h8_data(self, h8_file):
        h8_list = []
        start_time = time.time()
        for frame in range(len(h8_file)):
            with io.BytesIO(self.client.get(h8_file[frame])) as f:
                nc_data = np.load(f)
                h8_list.append(nc_data)
        data = np.array(h8_list)
        data[np.isnan(data)] = 0
        # import pdb
        # pdb.set_trace()
        if (not self.has_normed):
            data = self.norm_data(data.reshape(2,4,720,1120), data_type='h8')
        end_time=time.time()
        
        data = torch.from_numpy(data).float()
        return data

    def get_era5_data(self, era5_file):
        era5_data_surface = np.zeros((self.train_cfg.da_window_len, len(self.train_cfg.pred_names.surface), self.era5_size[0], self.era5_size[1]))
        if 'pressure' in self.pred_names.keys():
            era5_data_pressure = np.zeros((self.train_cfg.da_window_len, len(self.train_cfg.pred_names.pressure), self.era5_size[0], self.era5_size[1]))
        else:
            era5_data_pressure = []
        for type in range(len(era5_file)):
            if type == 0:
                v = 0
                t = 0
                for i in range(len(era5_file[type])): 
                    v_id = v % len(self.train_cfg.pred_names.surface)
                    v += 1
                    with io.BytesIO(self.client.get(era5_file[type][i])) as f:
                        data = np.load(f)[144:289, 320:545]
                        era5_data_surface[t, v_id, :, :] = data
                    if v_id == len(self.train_cfg.pred_names.surface) - 1:
                        t += 1
            if type == 1:
                v = 0
                t = 0
                for i in range(len(era5_file[type])): 
                    v_id = v % len(self.train_cfg.pred_names.pressure)
                    v += 1
                    with io.BytesIO(self.client.get(era5_file[type][i])) as f:
                        data = np.load(f)[144:289, 320:545]
                        era5_data_pressure[t, v_id, :, :] = data
                    if v_id == len(self.train_cfg.pred_names.pressure) - 1:
                        t += 1
        
        if (not self.has_normed):
            era5_data_surface[:,4,:,:]= era5_data_surface[:,4,:,:]*1000
            era5_data_surface = self.norm_data(era5_data_surface, data_type='pred')
            if 'pressure' in self.pred_names.keys():
                era5_data_pressure = self.norm_data(era5_data_pressure, data_type='pred')
        era5_data_surface = torch.from_numpy(era5_data_surface).float()
        if 'pressure' in self.pred_names.keys():
            era5_data_pressure = torch.from_numpy(era5_data_pressure).float()
        return era5_data_surface, era5_data_pressure

    def __len__(self):
        return len(self.pred_files)
        
    def get_input_files(self):
        h8_files = []
        h8_delta_t = []   # hour
        era5_files = []
        # start_time = self.start_time
        import tqdm
        for time_id in tqdm.tqdm(range(len(self.time_list))):
            init_time = self.time_list[time_id]
            
            # get h8 input file name
            h8_file, sub_h8_delta_t = self._get_h8_files(init_time=init_time, h8_sample_mode=self.train_cfg.h8_sample_mode, h8_frame_num=self.train_cfg.h8_frame_num)
            
            if len(h8_file) == 2:
                h8_files.append(h8_file)
                h8_delta_t.append(sub_h8_delta_t)
                
                # get era5 label file
                era5_file = self._get_era5_files(init_time=init_time, forecast_ahead_list=self.train_cfg.forecast_ahead_list, da_windown_len=self.train_cfg.da_window_len, pred_names=self.train_cfg.pred_names)
                era5_files.append(era5_file)  # [time[type[da_win*var_names]]]
            
        return h8_files, h8_delta_t, era5_files

    def _get_era5_files(self, init_time=None, forecast_ahead_list=[0,6], da_windown_len=6, pred_names=[]):
        sub_era5_file = []
        surface_tmp_list = []
        pressure_tmp_list = []
        init_time = init_time + pd.Timedelta(hours=forecast_ahead_list[0])
        for type in pred_names.keys():
            for time in range(da_windown_len):
                tmp_time = init_time + pd.Timedelta(hours=time)
                time_str = str(tmp_time.to_datetime64()).split('T')
                if type == 'surface':
                    for var_name in pred_names[type]:
                        url = f"{self.era5_data_path}single/{str(tmp_time.year)}/{time_str[0]}/{str(tmp_time.to_datetime64()).split('T')[1].split('.')[0]}-{var_name}.npy"
                        surface_tmp_list.append(url)
                elif type == 'pressure':
                    for var_name in pred_names[type]:
                        url = f"{self.era5_data_path}{str(tmp_time.year)}/{time_str[0]}/{str(tmp_time.to_datetime64()).split('T')[1].split('.')[0]}-{var_name}.npy"
                        pressure_tmp_list.append(url)
        sub_era5_file.append(surface_tmp_list)
        sub_era5_file.append(pressure_tmp_list)
        return sub_era5_file
    
    def _get_h8_files(self, init_time=None, h8_sample_mode='random', h8_frame_num=1):
        # todo 支持多frame h8输入
        minute_list = [0, 10, 20, 30, 40, 50]
        
        sub_h8_files = []
        sub_h8_delta_t = []
        if type(h8_sample_mode) == type(1):
            pass
        elif h8_sample_mode == 'nearest':
            for frame in range(h8_frame_num):
                not_exist_flag = True  
                error_idx = 0  
                try_id = 0
                while not_exist_flag and try_id <= 10:
                    
                    try:
                        try_id += 1
                        minute = minute_list[frame+error_idx]
                        ref_time = init_time + pd.Timedelta(hours=self.train_cfg.forecast_ahead_list[0]) - pd.Timedelta(minutes=minute)
                        ref_time_str = str(ref_time.to_datetime64()).split('-')
                        ref_time_year = ref_time_str[0]
                        ref_time_month = ref_time_str[1]
                        ref_time_day = ref_time_str[2][:2]
                        ref_hour = ref_time_str[2][3:5]
                        ref_mimute = ref_time_str[2][6:8]
                        
                        delta_time_second = (ref_time - init_time).total_seconds()
                        h8_url = f"{self.h8_data_path}/{'jma'}/{'netcdf'}/{ref_time_year}{ref_time_month}/{ref_time_day}/NC_H08_{ref_time_year}{ref_time_month}{ref_time_day}_{ref_hour}{ref_mimute}_R21_FLDK.02401_02401.npy"
                        
                        with io.BytesIO(self.client.get(h8_url)) as f:
                            tmp_data = np.load(f)
                        del tmp_data
                        not_exist_flag = False
                        sub_h8_delta_t.append(delta_time_second)
                        sub_h8_files.append(h8_url)
                    except:
                        error_idx += 1
                        print(f"File {h8_url} not exist, try again!")
                
        elif h8_sample_mode == 'random':
            pass
        else:
            raise NotImplementedError
        return sub_h8_files, sub_h8_delta_t

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

                if data.shape[1] == 5:
                    for idx, var_name in enumerate(self.train_cfg.pred_names.surface):
                        # [np.newaxis,:,np.newaxis,np.newaxis]
                        mean = np.array(single_level_mean_std['mean'][var_name])
                        std = np.array(single_level_mean_std['std'][var_name])
                        data[:,idx,:,:] = (data[:,idx,:,:] - mean) / std
                else:
                    pass
            elif data_type.lower() == 'w2k':
                
                with open(self.pred_single_norm_path, mode='r') as f:
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
        

if __name__ == '__main__':
    config_path = '/mnt/petrelfs/liuzili/code/DA_RS/configs/config_downscale.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)
    dataset = DARS_INF_Downscale_Dataset(config.train_cfg, 'valid')
    dataloader=torch.utils.data.DataLoader(dataset, batch_size = 4, shuffle=False,
                                                                drop_last=True,num_workers=4,
                                                                pin_memory = True, prefetch_factor = 3, persistent_workers = True)
    for id, data in enumerate(dataloader):
        a = id
            
