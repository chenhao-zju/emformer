import os
import abc
import datetime
import random
import json

# import h5py
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F

from concurrent.futures import ThreadPoolExecutor
import multiprocessing


ALL_ATMOSPHERIC_VARS = (
    "potential_vorticity",
    "specific_rain_water_content",
    "specific_snow_water_content",
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
    "vertical_velocity",
    "vorticity",
    "divergence",
    "relative_humidity",
    "ozone_mass_mixing_ratio",
    "specific_cloud_liquid_water_content",
    "specific_cloud_ice_water_content",
    "fraction_of_cloud_cover",
)

TARGET_SURFACE_VARS = (
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_v_component_of_wind",
    "10m_u_component_of_wind",
    "total_precipitation_6hr",
)
TARGET_SURFACE_NO_PRECIP_VARS = (
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_v_component_of_wind",
    "10m_u_component_of_wind",
)
TARGET_ATMOSPHERIC_VARS = (
    "temperature",
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "specific_humidity",
)
TARGET_ATMOSPHERIC_NO_W_VARS = (
    "temperature",
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
)


# FEATURE_DICT = {'Z500': (5, 0), 'T850': (2, 4), 'U10': (-4, 0), 'T2M': (-5, 0)}
FEATURE_DICT = {'Z500': (5, 0), 'T850': (2, 4), 'U10': (-3, 0), 'T2M': (-2, 0)}


SIZE_DICT = {0.25: [721, 1440], 0.5: [360, 720], 1.4: [128, 256]}

# higher+ surface z*13+q*13+u*13+v*13+t*13
# surface_features = []    # 'tp1h'
surface_features = ['t2m', 'u10', 'v10', 'msl', 'sp']    # 'tp1h' ['t2m', 'u10', 'v10', 'msl', 'sp'] 


higher_features = ['z', 'q', 'u', 'v', 't']
pressure_level = [1000.0, 925.0, 850.0, 700.0, 600.0, 500.0, 400.0, 300.0, 250.0, 200.0, 150.0, 100.0, 50.0]

total_levels= [1000.,  975.,  950.,  925.,  900.,  875.,  850.,  825.,  800.,
                775.,  750.,  700.,  650.,  600.,  550.,  500.,  450.,  400.,
                350.,  300.,  250.,  225.,  200.,  175.,  150.,  125.,  100.,
                70.,   50.,   30.,   20.,   10.,    7.,    5.,    3.,    2.,  1.]

mapping_dict = [total_levels.index(i) for i in pressure_level]

XXYY_DICT = {0.25: [391, 503, 1177, 1409], 1.4: [69, 89, 209, 249]}


def get_xxyy(range_xy):
    global_sn = [-90, 90]
    global_we = [-180, 180]
    regional_sn = [7.5, 36]
    regional_we = [114, 172.5]

    regional_x0 = int( ( (regional_sn[0] - global_sn[0] ) / (global_sn[1] - global_sn[0] ) ) * range_xy[0] )
    regional_x1 = int( ( (regional_sn[1] - global_sn[0] ) / (global_sn[1] - global_sn[0] ) ) * range_xy[0] )
    regional_y0 = int( ( (regional_we[0] - global_we[0] ) / (global_we[1] - global_we[0] ) ) * range_xy[1] )
    regional_y1 = int( ( (regional_we[1] - global_we[0] ) / (global_we[1] - global_we[0] ) ) * range_xy[1] )

    return (regional_x0, regional_x1, regional_y0, regional_y1)


def get_datapath_from_date(start_date, idx):
    t0 = start_date
    t = t0 + datetime.timedelta(hours=idx)
    year = t.year
    month = t.month
    day = t.day
    hour = t.hour
    date_file_name = f'{year}/{year}-{str(month).zfill(2)}-{str(day).zfill(2)}/{str(hour).zfill(2)}:00:00-'
    return date_file_name, f'{year}-{str(month).zfill(2)}-{str(day).zfill(2)}-{str(hour).zfill(2)}'
    # static_file_name = f'{year}/{year}.npy'
    # return date_file_name, static_file_name


def get_global_regional_data(x, target_xy, xxyy):
    x = torch.tensor(x, dtype=torch.float32)

    global_x = F.interpolate(x, size=(128, 256), mode='bilinear')
    regional_x = F.interpolate(x, size=(112, 232), mode='bilinear')

    # if len(inputs_shape) == 3:
    global_x = global_x.squeeze()
    regional_x = regional_x.squeeze()

    return (global_x, regional_x)


def get_perlin_noise(shape, seed):
    import noise

    # Parameters
    scale = 10.0        # Scale of the noise
    octaves = 6         # Number of layers of detail
    persistence = 0.5   # Amplitude of each octave
    lacunarity = 2.0    # Frequency of each octave

    # Generate Perlin noise
    perlin_noise = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            perlin_noise[i][j] = noise.pnoise2(
                i / scale,
                j / scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=shape[0],
                repeaty=shape[1],
                base=seed  # Random seed
            )

    return perlin_noise



class Era5Data(Dataset):
    def __init__(self,
                 data_params,
                 run_mode='train'):

        super(Era5Data, self).__init__()
        none_type = type(None)
        self.root_dir = data_params['root_dir']
        self.root_surface_dir = os.path.join(self.root_dir, "single")
        self.climate_dir = data_params['root_dir'] + 'climate_mean_day_128x256/1993-2016/'
        self.climate_surface_dir = data_params['root_dir'] + 'single/climate_mean_day_128x256/1993-2016/'

        self.statistic_dir = os.path.join(self.root_dir, "statistic")

        self._get_statistic()

        self.run_mode = run_mode
        self.t_in = data_params['t_in']
        self.h_size = data_params['h_size']
        self.w_size = data_params['w_size']
        self.ori_h_size = data_params['ori_h_size']
        self.ori_w_size = data_params['ori_w_size']
        self.data_frequency = data_params['data_frequency']
        self.valid_interval = data_params['valid_interval'] * self.data_frequency
        self.test_interval = data_params['test_interval'] * self.data_frequency
        self.train_interval = data_params['train_interval'] * self.data_frequency
        self.pred_lead_time = data_params['pred_lead_time']
        self.train_period = data_params['train_period']
        self.valid_period = data_params['valid_period']
        self.test_period = data_params['test_period']
        self.feature_dims = data_params['feature_dims']
        self.output_dims = data_params['feature_dims']
        self.surface_feature_size = data_params['surface_feature_size']
        if data_params['pressure_level_num'] != 0:
            self.level_feature_size = (self.feature_dims -
                                    self.surface_feature_size) // data_params['pressure_level_num']
        else:
            self.level_feature_size = 0
        self.patch = data_params['patch']
        if self.patch:
            self.patch_size = data_params['patch_size']

        self.globalregion = data_params['globalregion']
        if self.globalregion:
            grid_resolution = data_params['grid_resolution']
            self.xxyy = XXYY_DICT[grid_resolution]

        self.ens_test = data_params['ens_test']
        self.add_noise = data_params['add_noise']
        self.noise_weight = 0.1

        self.executor = ThreadPoolExecutor(max_workers=15)
        self.ori_h_size0 = self.ori_h_size - self.ori_h_size % self.patch_size



        if run_mode == 'train':
            self.t_out = data_params['t_out_train']
            self.interval = self.train_interval
            self.start_date = datetime.datetime(self.train_period[0], 1, 1, 0, 0, 0)

        elif run_mode == 'valid':
            self.t_out = data_params['t_out_valid']
            self.interval = self.valid_interval
            self.start_date = datetime.datetime(self.valid_period[0], 1, 1, 0, 0, 0)

        else:
            self.t_out = data_params['t_out_test']
            self.interval = self.test_interval
            self.start_date = datetime.datetime(self.test_period[0], 1, 1, 0, 0, 0)
            
    def __len__(self):
        if self.run_mode == 'train':
            self.train_len = self._get_file_count(self.root_dir, self.train_period)
            length = (self.train_len * self.data_frequency -
                      (self.t_out + self.t_in) * self.pred_lead_time) // self.train_interval

        elif self.run_mode == 'valid':
            self.valid_len = self._get_file_count(self.root_dir, self.valid_period)
            length = (self.valid_len * self.data_frequency -
                      (self.t_out + self.t_in) * self.pred_lead_time) // self.valid_interval

        else:
            self.test_len = self._get_file_count(self.root_dir, self.test_period)
            length = (self.test_len * self.data_frequency -
                      (self.t_out + self.t_in) * self.pred_lead_time) // self.test_interval

        if self.ens_test:
            length = 10
        
        return length

    def __getitem__(self, idx):
        inputs_lst = []
        label_lst = []
        self.months = []

        idx = idx * self.interval
        if self.run_mode != 'train':
            self.climate_lst = []
            self.climate_surface_lst = []

        if self.add_noise:
            random = np.random.randint(0, high=50000, size=None, dtype=int)
            self.noise = get_perlin_noise(shape=(self.ori_h_size0, self.ori_w_size), seed=idx+random)

        for t in range(self.t_in):
            cur_input_data_idx = idx + t * self.pred_lead_time
            half_path, date_time = get_datapath_from_date(self.start_date, cur_input_data_idx)
            # print('half data path: ', half_path, 'date_time: ', date_time)

            month = date_time.split('-')[1]
            month_embed = np.zeros([1, 12])
            month_embed[0, int(month)-1] = 1
            self.months.append(month_embed)

            x = self._get_weather_data(half_path)
            x = self._normalize(x)
            inputs_lst.append(x)

        self.months = [self.months[-1]]

        for t in range(self.t_out):
            cur_label_data_idx = idx + (self.t_in + t) * self.pred_lead_time
            label_path, date_time = get_datapath_from_date(self.start_date, cur_label_data_idx)
            label = self._get_weather_data(label_path)
            # print('label data path: ', label_path, 'date_time: ', date_time)

            label = self._normalize(label)
            label_lst.append(label)

            if self.run_mode != 'train':
                if '02-29' in date_time:  date_time = date_time.replace('02-29', '02-28')
                climate_features = self._get_climate_data(date_time)
                self.climate_lst.append(climate_features)


        x = np.stack(inputs_lst, axis=0).astype(np.float32)    # [t,h,w,level,feature]
        label = np.stack(label_lst, axis=0).astype(np.float32)

        if self.run_mode != 'train':
            self.climate = np.stack(self.climate_lst, axis=0).astype(np.float32)
            # self.climate = self.climate_features.transpose((0, 3, 1, 2))
            self.climate = np.squeeze( self.climate )
        return self._process_fn(x, label)


    def load_file(self, file_info):
        vdata = np.load(file_info)
        # vdata = torch.tensor(vdata[None, :, :])
        vdata = vdata.reshape([1, self.ori_h_size, self.ori_w_size])
        return vdata

    def _get_weather_data(self, half_path):
        all_level_paths = []
        for _, i in enumerate(higher_features):
            for j in pressure_level:
                # print(self.root_dir, half_path, i, str(j) )
                all_level_paths.append(self.root_dir+'/'+half_path + i +'-'+str(j)+'.npy')
        for i in surface_features:
            all_level_paths.append(self.root_surface_dir + '/' + half_path + i + '.npy')

        results = list(self.executor.map(self.load_file, all_level_paths))
        # results = self.pool.map(self.load_file, all_level_paths)

        input_initial_field = np.concatenate(results, axis=0)

        # print('all feature:', input_initial_field.shape )
        return input_initial_field

    def _get_climate_data(self, date_time):
        date_list = date_time.split('-')[1:-1]
        path_name = '-'.join(date_list)

        all_level_paths = []
        for i in higher_features:
            for j in pressure_level:
                all_level_paths.append( self.climate_dir + path_name + '/' + str(i) +'-'+str(j)+'.npy' )
        for i in surface_features:
            all_level_paths.append( self.climate_surface_dir + path_name + '/' + i + '.npy' )

        results = list(self.executor.map(self.load_file, all_level_paths))
        # results = self.pool.map(self.load_file, all_level_paths)

        input_initial_field = np.concatenate(results, axis=0)

        # print('all climate feature:', input_initial_field.shape )
        return input_initial_field

    @staticmethod
    def _get_file_count(path, period):
        count = 0
        for i in range(period[0], period[1]+1, 1):
            subpath = os.path.join(path, str(i))
            if os.path.exists(subpath):
                tmp_lst = os.listdir(subpath)
                count += 24*len(tmp_lst)

        return count

    def _get_statistic(self):
        self.surface_path = os.path.join(self.statistic_dir, 'mean_std_single.json')
        fs = open(self.surface_path, mode='r')
        self.mean_std_surface = json.load(fs)
        fs.close()

        self.level_path = os.path.join(self.statistic_dir, 'mean_std.json')
        fl = open(self.level_path, mode='r')
        self.mean_std_level = json.load(fl)
        fl.close()

        self.all_mean_surface = self.mean_std_surface['mean']
        self.all_mean_level = self.mean_std_level['mean']
        self.all_std_surface = self.mean_std_surface['std']
        self.all_std_level = self.mean_std_level['std']
        
        self.mean_surface = [self.all_mean_surface[i] for i in surface_features]
        self.std_surface = [self.all_std_surface[i] for i in surface_features]
        self.mean_surface = np.array(self.mean_surface)
        self.std_surface = np.array(self.std_surface)

        self.mean_pressure_level = [[self.all_mean_level[i][j] for j in mapping_dict] for i in higher_features]
        self.std_pressure_level = [[self.all_std_level[i][j] for j in mapping_dict] for i in higher_features]

        self.mean_pressure_level = np.array(self.mean_pressure_level).reshape(-1)
        self.std_pressure_level = np.array(self.std_pressure_level).reshape(-1)

        self.mean = np.concatenate([self.mean_pressure_level, self.mean_surface], axis=-1)[:, None, None]
        self.std = np.concatenate([self.std_pressure_level, self.std_surface], axis=-1)[:, None, None]



    def _normalize(self, x):
        x = (x - self.mean) / self.std
        return x

    def _process_fn(self, x, label):
        '''process_fn'''

        inputs = x[:, :, :self.ori_h_size0 , ...]
        labels = label[:, :, :self.ori_h_size0 , ...]
        
        inputs = np.squeeze(inputs)
        labels = np.squeeze(labels)

        months = np.concatenate(self.months, axis=0)
        months = np.squeeze(months)
        months = torch.tensor(months)

        # print(inputs.shape, labels.shape, self.climate.shape, months)

        if self.add_noise:
            m, _, _ = inputs.shape
            noises = np.tile(self.noise[None, ], (m, 1, 1))
            inputs += self.noise_weight * noises

        if self.run_mode == 'train':
            return inputs, labels, months
        else:
            return inputs, labels, self.climate, months
            

    def _patch(self, x, img_size, patch_size, output_dims):
        """ Partition the data into patches. """
        if self.run_mode == 'train':
            x = x.transpose(0, 2, 3, 1)
            h, w = img_size[0] // patch_size, img_size[1] // patch_size
            x = x.reshape(x.shape[0], h, patch_size, w, patch_size, output_dims)
            x = x.transpose(0, 1, 3, 2, 4, 5)
            x = np.squeeze(x.reshape(x.shape[0], h * w, patch_size * patch_size * output_dims))
        else:
            x = x.transpose(1, 0, 2, 3)
        return x


def get_data_loader_npy(params, distributed, run_mode):
    dataset = Era5Data(params, run_mode)
    sampler = DistributedSampler(dataset, shuffle=False) if distributed else None
    
    dataloader = DataLoader(dataset,
                            batch_size=int(params['batch_size']) if run_mode=='train' else 1,
                            num_workers=params['num_data_workers'] if run_mode=='train' else 1,
                            shuffle=False, #(sampler is None),
                            sampler=sampler if run_mode=='train' else None,
                            drop_last=True,
                            # pin_memory=torch.cuda.is_available()
                            pin_memory=False
                            )

    if run_mode=='train':
        return dataloader, dataset, sampler
    else:
        return dataloader, dataset


if __name__ == '__main__':
    data_params = {
            'name': 'era5',
            'root_dir': './era5_np128x256/',
            'feature_dims': 70,   # 68, 69, 
            't_in': 2,
            't_out_train': 1,
            't_out_valid': 1,
            't_out_test': 1,
            'valid_interval': 12,
            'test_interval': 12,
            'train_interval': 6,
            'pred_lead_time': 6,
            'data_frequency': 1,
            'train_period': [1990, 1990],   # [1979, 2020]
            'valid_period': [2021, 2021],
            'test_period': [2021, 2021],
            'patch': True,
            'patch_size': 4,
            'batch_size': 1,
            'num_data_workers': 1,
            'grid_resolution': 1.4,
            'h_size': 64,
            'w_size': 128,
            'surface_feature_size': 5,
            'pressure_level_num': 13,        # 13
            'globalregion': False,
            'ori_h_size': 128,
            'ori_w_size': 256,
            'ens_test': False,
            'add_noise': False,
        }
        
    from tqdm import tqdm

    dataloader, dataset, _ = get_data_loader_npy(data_params, False, 'train')

    for idx, data in tqdm(enumerate(dataloader)):

        inputs, targets, month = data
        print(inputs.shape, targets.shape)
        diff0 = torch.diff(inputs, dim=1)[:, 0]
        diff1 = inputs[:, 1] - inputs[:, 0]
        print(diff0[0, 0, 64] - diff1[0, 0, 64])
        if idx == 2: exit()






