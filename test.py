import os
import time
import copy
import argparse
import numpy as np
import torch
import math
import logging
from utils import logging_utils
from utils.YParams import YParams

from utils.data_loader_npyfiles import get_data_loader_npy, FEATURE_DICT, SIZE_DICT, surface_features, higher_features, pressure_level, XXYY_DICT
from utils.weighted_acc_rmse import weighted_rmse_torch_channels, weighted_acc_torch_channels
from networks import OneForecast, GraphCast, EMTransformerCast
from inference import WeatherForecast

logging_utils.config_logger()

def load_model(model, checkpoint_file):
    model.zero_grad()
    checkpoint_fname = checkpoint_file
    checkpoint = torch.load(checkpoint_fname)

    new_state_dict = {}
    for key, val in checkpoint['model_state'].items():
        name = key[7:]
        new_state_dict[name] = val  
    model.load_state_dict(new_state_dict, strict=False)
    return model

class InferenceModule(WeatherForecast):
    """
    Perform multiple rounds of model inference.
    """

    def __init__(self, model, config, dataset, run_mode='test', device='cpu', xxyy=None):
        super(InferenceModule, self).__init__(model, config, dataset, run_mode)
        self.dataset = dataset
        self.device = device

        self.total_model = config['total_model']


        self.mean_all = self.dataset.mean[:, 0, 0]
        self.std_all = self.dataset.std[:, 0, 0]
        
        self.feature_dims = config['feature_dims']
        self.use_moe = config['use_moe']
        self.add_kv = config['add_kv']

        if self.use_moe == 'densemoe' or self.use_moe == 'channelmoe' or self.use_moe=='channelmoev1' or self.use_moe=='channelmoev3':
            self.posembed = self.get_position()
            self.posembed = self.posembed.to(self.device, dtype = torch.float)

        self.xxyy = xxyy
        self.name = config['name']


        self.std_all = torch.tensor(self.std_all, dtype=torch.float32, device=self.device)[:, None, None]
        self.mean_all = torch.tensor(self.mean_all, dtype=torch.float32, device=self.device)[:, None, None]


    def forecast(self, inputs, labels):
        # unweighted_rmse_lst = []
        rmse_lst = []
        acc_lst = []
        kv_caches = None

        self.model.eval()
        with torch.no_grad():
            for i, t in enumerate( range(self.t_out_test) ):
                if self.total_model == 'oneforecast' or self.total_model == 'graphcast' or self.total_model == 'fuxi':
                    pred = self.model(inputs)

                elif self.total_model == 'emformer':
                    pred, _ = self.model(inputs, labels[:, t])

                label = labels[:, t]

                rmse = weighted_rmse_torch_channels(pred , label)

                denormalized_pred = pred * self.std_all + self.mean_all
                denormalized_label = label * self.std_all + self.mean_all
                denormalized_pred = denormalized_pred - self.climate[:, t]
                denormalized_label = denormalized_label - self.climate[:, t]

                acc = weighted_acc_torch_channels(denormalized_pred, denormalized_label)

                rmse_lst.append(rmse)
                acc_lst.append(acc)

                inputs = pred


        total_rmse = torch.mean(torch.stack(rmse_lst, dim=-1), dim=0)
        total_acc = torch.mean(torch.stack(acc_lst, dim=-1), dim=0)


        return total_rmse, total_acc

    
    def eval(self, data_loader, idx):
        logging.info("================================Start Evaluation================================")

        data_length = 0
        lat_weight_rmse = torch.zeros((self.config['feature_dims'], self.t_out_test), dtype=torch.float32, device=self.device)
        lat_weight_acc = torch.zeros((self.config['feature_dims'], self.t_out_test), dtype=torch.float32, device=self.device)


        for i, data in enumerate(data_loader):
            inputs, labels, climate, months  = map(lambda x: x.to(self.device, dtype = torch.float), data)
            self.months = months
            self.climate = climate

            lat_weight_rmse_step, lat_weight_acc_step = self._get_metrics(inputs, labels)

            if data_length == 0:
                lat_weight_rmse = lat_weight_rmse_step
                lat_weight_acc = lat_weight_acc_step
            else:
                lat_weight_rmse += lat_weight_rmse_step
                lat_weight_acc += lat_weight_acc_step

            data_length += 1

        logging.info(f'test dataset size: {data_length}')
        lat_weight_acc = (lat_weight_acc / data_length).cpu().numpy()
        lat_weight_rmse = (lat_weight_rmse / data_length).cpu().numpy()

        denormalized_lat_weight_rmse = lat_weight_rmse * self.total_std[:, None]

        if self.config["save_rmse_acc"]:
            if self.xxyy != None:
                np.save(os.path.join(self.config['experiment_dir'],
                                    f"ens_global_{self.name}_rmse_step{str(idx)}.npy"), denormalized_lat_weight_rmse)
                np.save(os.path.join(self.config['experiment_dir'],
                                    f"ens_global_{self.name}_acc_step{str(idx)}.npy"), lat_weight_acc)
            else:
                np.save(os.path.join(self.config['experiment_dir'],
                                    f"denormalized_lat_normalized_rmse_step{str(idx)}.npy"), lat_weight_rmse)
                np.save(os.path.join(self.config['experiment_dir'],
                                    f"denormalized_lat_weight_rmse_step{str(idx)}.npy"), denormalized_lat_weight_rmse)
                np.save(os.path.join(self.config['experiment_dir'],
                                    f"lat_weight_acc_step{str(idx)}.npy"), lat_weight_acc)

        self._print_key_metrics(denormalized_lat_weight_rmse, lat_weight_acc)

        logging.info("================================End Evaluation================================")
        return denormalized_lat_weight_rmse, lat_weight_acc

    

    def eval_longtimes(self, data_loader, steps):
        '''
        Eval the model using test dataset or validation dataset.

        Args:
            dataset: The dataset for eval, including inputs and labels.
        '''
        logging.info("================================Start Evaluation================================")

        for i, data in enumerate(data_loader):
            inputs, labels, months  = map(lambda x: x.to(self.device, dtype = torch.float), data)
            self.months = months

            if i>0:
                inputs = pred

            self.model.eval()
            with torch.no_grad():
                if self.total_model == 'skno':
                    pred, _, _ = self.model(inputs, target=labels, months=self.months, embed_layers=None)
                elif self.total_model == 'oneforecast' or self.total_model == 'graphcast' or self.total_model == 'fuxi':
                    pred = self.model(inputs)

            if i == steps-1:
                pred = pred.cpu().numpy()
                labels = labels.cpu().numpy()
                break

        np.save(os.path.join(self.config['experiment_dir'], f"pred.npy"), pred)
        np.save(os.path.join(self.config['experiment_dir'], f"label.npy"), labels)

        logging.info("================================End Evaluation================================")


    def _get_metrics(self, inputs, labels):
        """Get lat_weight_rmse and lat_weight_acc metrics"""
        total_rmse, total_acc = self.forecast(inputs, labels)

        return total_rmse, total_acc

    def get_position(self):
        num_feature = len(higher_features)
        num_level = len(pressure_level)
        num_surface = len(surface_features)

        if num_surface > 0:
            assert self.config['num_exports'] == num_feature+1, 'num_expert should be equal to num_feature + 1'
        else:
            assert self.config['num_exports'] == num_feature, 'num_expert should be equal to num_feature'

        inputs = torch.zeros([self.config['num_exports'], num_level * num_feature + num_surface])
        for i in range(num_feature):
            inputs[i, i*num_level:(i+1)*num_level] = torch.ones(num_level)
        if num_surface > 0:
            inputs[-1, -num_surface:] = torch.ones(num_surface)
        
        return inputs


    def _get_lat_weight(self):
        lat_t = np.arange(0, self.h_size)
        s = np.sum(np.cos(math.pi / 180. * self._lat(lat_t)))
        # self.h_size * np.cos(PI / 180. * self._lat(j)) / s
        weight = self._latitude_weighting_factor(lat_t, s)
        return weight

    def _calculate_lat_weighted_error(self, label, prediction):
        """calculate latitude weighted error"""
        weight = self._get_lat_weight()
        grid_node_weight = np.repeat(weight, self.w_size, axis=0).reshape(-1, 1)
        error = np.square(label - prediction) # the index 0 of label shape is batch_size
        # logging.info(f'error shape: {error.shape}, grid_node_weight shape: {grid_node_weight.shape}')
        lat_weight_error = np.sum(error * grid_node_weight, axis=2)
        lat_weight_error = np.sum(lat_weight_error, axis=0)
        return lat_weight_error

    def _calculate_lat_weighted_acc(self, label, prediction):
        """calculate latitude weighted acc"""
        prediction = prediction - self.climates
        label = label - self.climates

        prediction = prediction * self.std_all + self.mean_all
        label = label * self.std_all + self.mean_all
        weight = self._get_lat_weight()
        grid_node_weight = np.repeat(weight, self.w_size, axis=0).reshape(1, -1, 1)
        acc_numerator = np.sum(prediction * label * grid_node_weight, axis=2)
        acc_denominator = np.sqrt(np.sum(grid_node_weight * prediction ** 2,
                                         axis=2) * np.sum(grid_node_weight * label ** 2, axis=2))

        try:
            # acc = acc_numerator / acc_denominator
            acc = np.divide(acc_numerator, acc_denominator)
            acc = np.sum(acc, axis=0)
        except ZeroDivisionError as e:
            print(repr(e))
        return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", default='00', type=str)
    parser.add_argument("--yaml_config", default='./config/AFNO.yaml', type=str)
    parser.add_argument("--config", default='full_field', type=str)
    parser.add_argument("--override_dir", default=None, type = str, help = 'Path to store inference outputs; must also set --weights arg')
    parser.add_argument("--weights", default=None, type=str, help = 'Path to model weights, for use with override_dir option')
    
    args = parser.parse_args()
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params['world_size'] = 1
    # params['interp'] = args.interp
    params['global_batch_size'] = params.batch_size
    # params['global_batch_size'] = 32

    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True
    # vis = args.vis

    # Set up directory
    if args.override_dir is not None:
      assert args.weights is not None, 'Must set --weights argument if using --override_dir'
      expDir = args.override_dir
    else:
      assert args.weights is None, 'Cannot use --weights argument without also using --override_dir'
      expDir = os.path.join(params.exp_dir, args.config, str(args.run_num))

    if not os.path.isdir(expDir):
      os.makedirs(expDir)

    params['name'] = args.override_dir.split('/')[-2]
    print('log name: ', params['name'])

    params['experiment_dir'] = os.path.abspath(expDir)
    params['best_checkpoint_path'] = args.weights if args.override_dir is not None else os.path.join(expDir, 'training_checkpoints/best_ckpt.tar')
    params['resuming'] = False
    params['local_rank'] = 0

    params['surface_features'] = surface_features
    params['higher_features'] =  higher_features
    params['pressure_level'] = pressure_level

    params['old_surface_feature'] = [] 
    params['old_higher_features'] = ['z', 'q', 'u', 'v']
    params['old_pressure_level'] = [1000.0, 925.0, 850.0, 700.0, 600.0, 500.0, 400.0, 300.0, 250.0, 200.0, 150.0, 100.0, 50.0]

    params['xxyy'] = XXYY_DICT[ params['grid_resolution'] ]

    logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'global_inference_out_weighted.log'))
    logging_utils.log_versions()
    params.log()

    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    if params['total_model'] == 'oneforecast':
        model = OneForecast(input_dim_grid_nodes=params['feature_dims'], output_dim_grid_nodes=params['feature_dims'], input_res=(params['h_size'], params['w_size']) ).to(device)
    elif params['total_model'] == 'graphcast':
        model = GraphCast(input_dim_grid_nodes=params['feature_dims'], output_dim_grid_nodes=params['feature_dims'], input_res=(params['h_size'], params['w_size']) ).to(device)

    elif params['total_model'] == 'emformer':
        model = EMTransformerCast(dim = params['embed_dim'], depth = (2, (2, (2, (2, params['encoder_depths'], 2), 2), 2), 2), updown_sample_type = 'linear', in_chans = params['feature_dims'], out_chans = params['feature_dims'], H = params['h_size'], W = params['w_size'], patch_size = params['patch_size'], add_kv = params['add_kv']).to(device)

    checkpoint_file  = params['best_checkpoint_path']
    model = load_model(model, checkpoint_file)
    model = model.to(device)


    test_data_loader, test_dataset = get_data_loader_npy(params, False, run_mode='test')
    logging.info(f"Test dataset size: {len(test_dataset)}")

    start_time = time.time()
    inference_module = InferenceModule(model, params, test_dataset, run_mode='test', device=device)

    inference_module.eval(test_data_loader, 0)
    
    logging.info(f"End-to-End total time: {time.time() - start_time} s")
