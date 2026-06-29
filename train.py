import os
import time
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.amp as amp
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import logging
from utils import logging_utils
logging_utils.config_logger()
from utils.YParams import YParams

from utils.data_loader_npyfiles import (
    get_data_loader_npy,
    surface_features,
    higher_features,
    pressure_level,
)

from networks import OneForecast, GraphCast, EMTransformerCast

from utils.weighted_acc_rmse import weighted_rmse_torch
from networks.l2_loss import L2_LOSS
from collections import OrderedDict
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict

from test import InferenceModule

try:
    from apex import optimizers as apex_optimizers
except Exception:
    apex_optimizers = None

try:
    import wandb
except Exception:
    wandb = None

DECORRELATION_TIME = 36  # 9 days


class Trainer():
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def __init__(self, params, world_rank):
        self.params = params
        self.world_rank = world_rank
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

        if params.log_to_wandb and wandb is not None:
            wandb.init(config=params, name=params.name, group=params.group, project=params.project)

        logging.info('rank %d, begin data loader init' % world_rank)

        self.train_data_loader, self.train_dataset, self.train_sampler = get_data_loader_npy(params, dist.is_initialized(), run_mode='train')
        self.valid_data_loader, self.valid_dataset = get_data_loader_npy(params, dist.is_initialized(), run_mode='valid')

        self.loss_type = params['loss']
        self.loss_weight = params['loss_weight']

        logging.info(f'****** using {self.loss_type} in model training ******')
        if self.loss_type == 'l2':
            learn_log_variance = dict(flag=True, channels=params['feature_dims'], logvar_init=0., requires_grad=True)
            self.loss_gen = L2_LOSS(learn_log_variance=learn_log_variance).to(self.device)
            self.loss_recons = L2_LOSS(learn_log_variance=learn_log_variance).to(self.device)
        else:
            learn_log_variance = dict(flag=True, channels=params['feature_dims'], logvar_init=0., requires_grad=True)
            self.loss_gen = L2_LOSS(learn_log_variance=learn_log_variance).to(self.device)
            self.loss_recons = L2_LOSS(learn_log_variance=learn_log_variance).to(self.device)

        logging.info('rank %d, data loader initialized' % world_rank)

        self.use_moe = params['use_moe']
        self.use_cl = params['use_cl']
        self.mlp_ratio = params['mlp_ratio']

        self.globalregion = params['globalregion']
        self.add_kv = params['add_kv']

        self.total_model = params['total_model']

        self.surface_features = params['surface_features'] = surface_features
        self.higher_features = params['higher_features'] = higher_features
        self.pressure_level = params['pressure_level'] = pressure_level

        if params['total_model'] == 'oneforecast':
            self.model = OneForecast(input_dim_grid_nodes=params['feature_dims'], output_dim_grid_nodes=params['feature_dims'], input_res=(params['h_size'], params['w_size'])).to(self.device)
        elif params['total_model'] == 'graphcast':
            self.model = GraphCast(input_dim_grid_nodes=params['feature_dims'], output_dim_grid_nodes=params['feature_dims'], input_res=(params['h_size'], params['w_size'])).to(self.device)
        elif params['total_model'] == 'emformer':
            self.model = EMTransformerCast(dim=params['embed_dim'], depth=(2, (2, (2, (2, params['encoder_depths'], 2), 2), 2), 2), updown_sample_type='linear', in_chans=params['feature_dims'], out_chans=params['feature_dims'], H=params['h_size'], W=params['w_size'], patch_size=params['patch_size'], add_kv=self.add_kv).to(self.device)

        if params.log_to_wandb and wandb is not None:
            wandb.watch(self.model)

        if params.optimizer_type == 'FusedAdam' and apex_optimizers is not None:
            self.optimizer = apex_optimizers.FusedAdam(self.model.parameters(), lr=params.lr)
        elif params.optimizer_type == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=params.lr)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params.lr)

        if params.enable_amp == True:
            self.gscaler = amp.GradScaler("cuda")

        self.iters = 0
        self.startEpoch = 0
        if params.resuming:
            logging.info("Loading checkpoint %s" % params.checkpoint_path)
            with torch.no_grad():
                try:
                    logging.info("Loading checkpoint %s" % params.checkpoint_path)
                    self.restore_checkpoint(params.checkpoint_path)
                except:
                    logging.info("Loading checkpoint %s" % params.checkpoint)
                    self.restore_checkpoint(params.checkpoint)

        if dist.is_initialized():
            self.model = DistributedDataParallel(self.model.to(params.local_rank),
                                                 device_ids=[params.local_rank],
                                                 output_device=[params.local_rank], find_unused_parameters=False)

        self.epoch = self.startEpoch

        if params.scheduler == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.2, patience=5, mode='min')
        elif params.scheduler == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=params.max_epochs, last_epoch=self.startEpoch - 1, eta_min=params.min_lr)
        else:
            self.scheduler = None

        if params.log_to_screen:
            logging.info("Number of trainable model parameters: {}".format(self.count_parameters()))

        self.inference = InferenceModule(self.model, self.params, self.valid_dataset, run_mode='valid', device=self.device)

    def switch_off_grad(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def loss_obj(self, pred1, pred2, target1, target2):
        loss1 = self.loss_gen(pred1, target1)
        loss2 = self.loss_recons(pred2, target2)
        return (1 - self.loss_weight) * loss1 + self.loss_weight * loss2

    def train(self):
        if self.params.log_to_screen:
            logging.info("Starting Training Loop...")

        best_valid_loss = 1.e6
        for epoch in range(self.startEpoch, self.params.max_epochs):
            if dist.is_initialized():
                self.train_sampler.set_epoch(epoch)
            start = time.time()
            tr_time, data_time, train_logs = self.train_one_epoch()
            valid_time, valid_logs = self.validate_one_epoch()

            if self.params.scheduler == 'ReduceLROnPlateau':
                self.scheduler.step(valid_logs['valid_loss'])
            elif self.params.scheduler == 'CosineAnnealingLR':
                self.scheduler.step()
                if self.epoch >= self.params.max_epochs:
                    logging.info("Terminating training after reaching params.max_epochs while LR scheduler is set to CosineAnnealingLR")
                    exit()

            if self.params.log_to_wandb and wandb is not None:
                for pg in self.optimizer.param_groups:
                    lr = pg['lr']
                wandb.log({'lr': lr})

            if self.world_rank == 0:
                if self.params.save_checkpoint:
                    self.save_checkpoint(self.params.checkpoint_path)
                    if self.epoch % 5 == 0:
                        mid_checkpoint_path = self.params.mid_checkpoint_path + f'ckpt{str(self.epoch)}.tar'
                        os.system(f'cp {self.params.checkpoint_path} {mid_checkpoint_path}')

                        if len(os.listdir(self.params.mid_checkpoint_path)) >= 4:
                            name = mid_checkpoint_path.split('/')[-1]
                            num = int(name.split('.')[0][4:]) - 5
                            rm_checkpoint_path = self.params.mid_checkpoint_path + f'ckpt{str(num)}.tar'
                            os.system(f'rm -r {rm_checkpoint_path}')

                    if valid_logs['valid_loss'] <= best_valid_loss:
                        self.save_checkpoint(self.params.best_checkpoint_path)
                        best_valid_loss = valid_logs['valid_loss']

            if self.params.log_to_screen:
                logging.info('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
                current_lr = self.optimizer.param_groups[0]['lr']
                logging.info('Train loss: {}. Valid loss: {}. Learning Rate: {}.'.format(train_logs['loss'], valid_logs['valid_loss'], current_lr))
                logging.info(f"Test results of RMSE: z500: {valid_logs['z500']}, t2m: {valid_logs['t2m']}, t850: {valid_logs['t850']}, u10: {valid_logs['u10']}")

    def train_one_epoch(self):
        self.epoch += 1
        tr_time = 0
        data_time = 0
        self.model.train()

        data_start = time.time()

        for i, data in enumerate(self.train_data_loader):
            self.iters += 1

            if self.globalregion:
                (global_inp, regional_inp), (global_target, regional_target), months = data
                global_inp, regional_inp, global_target, regional_target, months = map(lambda x: x.to(self.device, dtype=torch.float32), [global_inp, regional_inp, global_target, regional_target, months])
            else:
                inp, target, months = map(lambda x: x.to(self.device, dtype=torch.float32), data)

            data_time += time.time() - data_start

            tr_start = time.time()
            kv_caches = None

            t_out_train = self.params['t_out_train']
            for j in range(t_out_train):
                if t_out_train == 1:
                    tar = target.clone()
                else:
                    if j > 0:
                        inp = gen.detach()
                    tar = target[:, j]

                self.model.zero_grad()
                with amp.autocast('cuda'):
                    if self.total_model == 'oneforecast' or self.total_model == 'graphcast':
                        gen = self.model(inp)
                        loss = self.loss_gen(gen, tar)
                    elif self.total_model == 'emformer':
                        gen, loss = self.model(inp, tar)

                if self.params.enable_amp:
                    self.gscaler.scale(loss).backward(retain_graph=True)
                    self.gscaler.step(self.optimizer)
                else:
                    loss.backward()
                    self.optimizer.step()

                if self.params.enable_amp:
                    self.gscaler.update()

                if self.params['use_grad_clip']:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)

                tr_time += time.time() - tr_start

        logs = {'loss': loss}

        if dist.is_initialized():
            for key in sorted(logs.keys()):
                dist.all_reduce(logs[key].detach())
                logs[key] = float(logs[key] / dist.get_world_size())

        if self.params.log_to_wandb and wandb is not None:
            wandb.log(logs, step=self.epoch)

        return tr_time, data_time, logs

    def validate_one_epoch(self):
        self.model.eval()

        valid_buff = torch.zeros((3), dtype=torch.float32, device=self.device)
        valid_loss = valid_buff[0].view(-1)
        valid_l1 = valid_buff[1].view(-1)
        valid_steps = valid_buff[2].view(-1)
        valid_weighted_rmse = torch.zeros((self.params.N_out_channels), dtype=torch.float32, device=self.device)

        valid_start = time.time()

        with torch.no_grad():
            for i, data in enumerate(self.valid_data_loader):
                inp, tar, climate, months = map(lambda x: x.to(self.device, dtype=torch.float32), data)
                tar = tar[:, 0] if tar.dim() == 5 else tar

                if self.total_model == 'oneforecast' or self.total_model == 'graphcast':
                    gen = self.model(inp)
                    valid_loss += self.loss_gen(gen, tar)
                elif self.total_model == 'emformer':
                    gen, loss = self.model(inp, tar)
                    valid_loss += loss

                valid_l1 += F.mse_loss(gen, tar)
                valid_steps += 1.
                valid_weighted_rmse += weighted_rmse_torch(gen, tar)

        if dist.is_initialized():
            dist.all_reduce(valid_buff)
            dist.all_reduce(valid_weighted_rmse)

        valid_buff[0:2] = valid_buff[0:2] / valid_buff[2]
        valid_weighted_rmse = valid_weighted_rmse / valid_buff[2]

        valid_buff_cpu = valid_buff.detach().cpu().numpy()
        valid_weighted_rmse_cpu = self.inference.total_std * valid_weighted_rmse.detach().cpu().numpy()

        valid_time = time.time() - valid_start

        num_surface_variables = len(self.surface_features)
        logs = {'valid_l1': valid_buff_cpu[1], 'valid_loss': valid_buff_cpu[0], 'z500': valid_weighted_rmse_cpu[5], 't2m': valid_weighted_rmse_cpu[-5], 't850': valid_weighted_rmse_cpu[54], 'u10': valid_weighted_rmse_cpu[-4]}

        for i, name in enumerate(self.surface_features):
            logs[name] = valid_weighted_rmse_cpu[i - num_surface_variables]

        if self.params.log_to_wandb and wandb is not None:
            wandb.log(logs, step=self.epoch)

        return valid_time, logs

    def save_checkpoint(self, checkpoint_path, model=None):
        if not model:
            model = self.model

        torch.save({'iters': self.iters, 'epoch': self.epoch, 'model_state': model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()}, checkpoint_path)

    def restore_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cuda:{}'.format(self.params.local_rank))
        new_state_dict = OrderedDict()
        for key, val in checkpoint['model_state'].items():
            name = key[7:]
            new_state_dict[name] = val
        self.model.load_state_dict(new_state_dict, strict=False)

        if self.params['checkpoint'] == '':
            self.iters = checkpoint['iters']
            self.startEpoch = checkpoint['epoch']

        if self.params.resuming:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", default='00', type=str)
    parser.add_argument("--yaml_config", default='./config/EMFormer.yaml', type=str)
    parser.add_argument("--checkpoint", default='', type=str)
    parser.add_argument("--exp_dir", default='./logs/test', type=str)
    parser.add_argument("--config", default='EMFormer', type=str)
    parser.add_argument("--enable_amp", action='store_true')
    parser.add_argument("--epsilon_factor", default=0, type=float)
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument("--num_data_workers", default=-1, type=int)

    args = parser.parse_args()

    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params['epsilon_factor'] = args.epsilon_factor
    if args.num_data_workers >= 0:
        params['num_data_workers'] = args.num_data_workers

    params['world_size'] = 1
    if 'WORLD_SIZE' in os.environ:
        params['world_size'] = int(os.environ['WORLD_SIZE'])
    logging.info(f"world_size:, {params['world_size']}")

    if wandb is not None:
        wandb.require("core")
    world_rank = 0
    local_rank = 0
    if params['world_size'] > 1:
        dist.init_process_group(backend='nccl',
                                init_method='env://',
                                )
        local_rank = int(os.environ["LOCAL_RANK"])
        args.gpu = local_rank
        world_rank = dist.get_rank()
        params['global_batch_size'] = params.batch_size
        params['batch_size'] = int(params.batch_size // params['world_size'])

    logging.info(f"local_rank:, {local_rank}")

    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = True

    expDir = args.exp_dir + args.config + '/' + str(args.run_num) + '/'
    if world_rank == 0:
        if not os.path.isdir(expDir):
            os.makedirs(expDir)
            os.makedirs(expDir + 'training_checkpoints/')

    params['experiment_dir'] = os.path.abspath(expDir)
    params['checkpoint_path'] = expDir + 'training_checkpoints/ckpt.tar'
    params['mid_checkpoint_path'] = expDir + 'training_checkpoints/'
    params['best_checkpoint_path'] = expDir + 'training_checkpoints/best_ckpt.tar'

    params['checkpoint'] = args.checkpoint

    args.resuming = True if os.path.isfile(params.checkpoint_path) or os.path.isfile(params.checkpoint) else False

    params['resuming'] = args.resuming
    params['local_rank'] = local_rank
    params['enable_amp'] = args.enable_amp

    params['name'] = args.config + '_' + str(args.run_num)
    params['group'] = "emformer"
    params['project'] = "emformer"
    if world_rank == 0:
        logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'out.log'))
        logging_utils.log_versions()
        params.log()

    params['log_to_wandb'] = (world_rank == 0) and params['log_to_wandb']
    params['log_to_screen'] = (world_rank == 0) and params['log_to_screen']

    params['in_channels'] = np.array(list(range(params['in_channels'])))
    params['out_channels'] = np.array(list(range(params['out_channels'])))
    if params.orography:
        params['N_in_channels'] = len(params['in_channels']) + 1
    else:
        params['N_in_channels'] = len(params['in_channels'])
    params['N_out_channels'] = len(params['out_channels'])

    if world_rank == 0:
        hparams = ruamelDict()
        yaml = YAML()
        for key, value in params.params.items():
            hparams[str(key)] = str(value)
        with open(os.path.join(expDir, 'hyperparams.yaml'), 'w') as hpfile:
            yaml.dump(hparams, hpfile)

    trainer = Trainer(params, world_rank)
    trainer.train()
    logging.info('DONE ---- rank %d' % world_rank)