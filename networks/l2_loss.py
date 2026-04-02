import torch.nn as nn
import torch.nn.functional as F
import torch
# from nwp.registry import MODELS
from typing import Dict, List
# from .utils import weight_reduce_loss
from mmengine.dist import get_rank
from collections import deque
from mmengine.logging import print_log


from utils.weighted_acc_rmse import lat, latitude_weighting_factor_torch, weighted_rmse_torch      # , weighted_rmse_torch_channels


def weighted_rmse_torch_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each chann
    num_lat = pred.shape[2]
    #num_long = target.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)

    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor_torch(lat_t, num_lat, s), (1, 1, -1, 1))
    result = weight * (pred - target)**2.
    return result


class L2_LOSS(nn.Module):
    """Cross entropy loss.
    Args:
        use_sigmoid (bool): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_soft (bool): Whether to use the soft version of CrossEntropyLoss.
            Defaults to False.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to 'mean'.
        loss_weight (float):  Weight of the loss. Defaults to 1.0.
        class_weight (List[float], optional): The weight for each class with
            shape (C), C is the number of classes. Default None.
        pos_weight (List[float], optional): The positive weight for each
            class with shape (C), C is the number of classes. Only enabled in
            BCE loss when ``use_sigmoid`` is True. Default None.
    """

    def __init__(self,
                 reduction='mean',
                 loss_weight=None,
                 learn_log_variance: Dict= dict(flag=False, requires_grad=True, channels=69, logvar_init=0.),
                 normalized_weight_cfg: Dict = dict(flag=False, queue_length=1000., channels=69, norm_value=1.),
                 class_weight: List=None,
                 pos_weight: List = None):
        super(L2_LOSS, self).__init__()
        self.learn_log_variance =  learn_log_variance.get('flag')
        self.normlized_weight = normalized_weight_cfg.get('flag')
        assert not (
                self.learn_log_variance and self.normlized_weight
        ), 'learn_log_variance and normlized_weight could not be set simultaneously'

        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.pos_weight = pos_weight
        self.normalized_weight_cfg = normalized_weight_cfg
        self.logvar = None
        
        if self.learn_log_variance:
            self.channels = learn_log_variance.get('channels')
            self.logvar = nn.Parameter(
                torch.ones(size=(1,self.channels,1,1),requires_grad=learn_log_variance.get('requires_grad')) * learn_log_variance.get('logvar_init')
                )
        if self.normlized_weight:
            self.channels = normalized_weight_cfg.get('channels')
            self.queues = [deque(maxlen=normalized_weight_cfg.get('queue_length')) for _ in range(self.channels)]
            for i in range(self.channels):
                self.queues[i].append(1.)
            self.logvar = torch.ones(size=(1,self.channels))
        
        self.count = 0
    def forward(self,
                pred,
                label,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        # only BCE loss has pos_weight
        if self.pos_weight is not None and self.use_sigmoid:
            pos_weight = pred.new_tensor(self.pos_weight)
            kwargs.update({'pos_weight': pos_weight})
        else:
            pos_weight = None

        loss = torch.square(pred-label)
        if self.learn_log_variance or self.normlized_weight:
          
            if self.learn_log_variance:
                assert self.logvar.data.ndim == loss.ndim
                loss = loss / (torch.exp(self.logvar)) + self.logvar
                loss = loss.mean(dim=(-1,-2))

                if get_rank() == 0:
                    self.count+=1
                    if self.count%100==0:
                        print_log(f'loss channel weight:{self.logvar.data.squeeze()}', logger='current')
                return loss.mean()
            
            if self.normlized_weight:
                loss = loss.mean(dim=(-1,-2))
                for i in range(self.channels):

                    self.queues[i].append(loss[:,i].mean().item())
                    # self.normalized_weight_cfg.get('norm_value')
                    weight = loss.detach().mean().item()/(10e-9+torch.tensor(list(self.queues[i])).mean())
                    self.logvar[:,i] = weight
                
                assert self.logvar.data.ndim == loss.ndim     
                loss = loss * self.logvar.to(loss.device)
                
                if get_rank() == 0:
                    self.count+=1
                    if self.count%100==0:
                        print(self.logvar.data.squeeze())
                        print(len(self.queues[i]))
                        
                return loss, (1/self.logvar).mean()
        else:
            if self.loss_weight is not None:
                loss = torch.tensor(self.loss_weight).view(1, -1,1 ,1 ).to(loss.device)*loss

        loss = loss.mean()

        return loss


class LatWeighted_L2_LOSS(nn.Module):
    """Cross entropy loss.
    Args:
        use_sigmoid (bool): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_soft (bool): Whether to use the soft version of CrossEntropyLoss.
            Defaults to False.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to 'mean'.
        loss_weight (float):  Weight of the loss. Defaults to 1.0.
        class_weight (List[float], optional): The weight for each class with
            shape (C), C is the number of classes. Default None.
        pos_weight (List[float], optional): The positive weight for each
            class with shape (C), C is the number of classes. Only enabled in
            BCE loss when ``use_sigmoid`` is True. Default None.
    """

    def __init__(self,
                 reduction='mean',
                 loss_weight=None,
                 learn_log_variance: Dict= dict(flag=False, requires_grad=True, channels=69, logvar_init=0.),
                 normalized_weight_cfg: Dict = dict(flag=False, queue_length=1000., channels=69, norm_value=1.),
                 class_weight: List=None,
                 pos_weight: List = None):
        super(LatWeighted_L2_LOSS, self).__init__()
        self.learn_log_variance =  learn_log_variance.get('flag')
        self.normlized_weight = normalized_weight_cfg.get('flag')
        assert not (
                self.learn_log_variance and self.normlized_weight
        ), 'learn_log_variance and normlized_weight could not be set simultaneously'

        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.pos_weight = pos_weight
        self.normalized_weight_cfg = normalized_weight_cfg
        self.logvar = None
        
        if self.learn_log_variance:
            self.channels = learn_log_variance.get('channels')
            self.logvar = nn.Parameter(
                torch.ones(size=(1,self.channels,1,1),requires_grad=learn_log_variance.get('requires_grad')) * learn_log_variance.get('logvar_init')
                )
        if self.normlized_weight:
            self.channels = normalized_weight_cfg.get('channels')
            self.queues = [deque(maxlen=normalized_weight_cfg.get('queue_length')) for _ in range(self.channels)]
            for i in range(self.channels):
                self.queues[i].append(1.)
            self.logvar = torch.ones(size=(1,self.channels))
        
        self.count = 0

    def forward(self,
                pred,
                label,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        # only BCE loss has pos_weight
        if self.pos_weight is not None and self.use_sigmoid:
            pos_weight = pred.new_tensor(self.pos_weight)
            kwargs.update({'pos_weight': pos_weight})
        else:
            pos_weight = None

        # loss = torch.square(pred-label)
        loss = weighted_rmse_torch_channels(pred, label)

        if self.learn_log_variance or self.normlized_weight:
          
            if self.learn_log_variance:
                assert self.logvar.data.ndim == loss.ndim
                loss = loss / (torch.exp(self.logvar)) + self.logvar
                loss = loss.mean(dim=(-1,-2))

                if get_rank() == 0:
                    self.count+=1
                    if self.count%100==0:
                        print_log(f'loss channel weight:{self.logvar.data.squeeze()}', logger='current')
                return loss.mean()
            
            if self.normlized_weight:
                # loss = loss.mean(dim=(-1,-2))
                for i in range(self.channels):

                    self.queues[i].append(loss[:,i].mean().item())
                    # self.normalized_weight_cfg.get('norm_value')
                    weight = loss.detach().mean().item()/(10e-9+torch.tensor(list(self.queues[i])).mean())
                    self.logvar[:,i] = weight
                
                assert self.logvar.data.ndim == loss.ndim     
                loss = loss * self.logvar.to(loss.device)
                
                if get_rank() == 0:
                    self.count+=1
                    if self.count%100==0:
                        print(self.logvar.data.squeeze())
                        print(len(self.queues[i]))
                        
                return loss, (1/self.logvar).mean()
        else:
            if self.loss_weight is not None:
                loss = torch.tensor(self.loss_weight).view(1, -1,1 ,1 ).to(loss.device)*loss

        loss = loss.mean()

        return loss


def weights_torch_channels(img_size=(128, 256)):
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each chann
    num_lat = img_size[0]
    #num_long = target.shape[2]
    lat_t = torch.arange(start=0, end=num_lat)

    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor_torch(lat_t, num_lat, s), (1, 1, -1, 1))
    return weight


class LatWeighted_L2_LOSSv2(nn.Module):
    """Cross entropy loss.
    Args:
        use_sigmoid (bool): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_soft (bool): Whether to use the soft version of CrossEntropyLoss.
            Defaults to False.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to 'mean'.
        loss_weight (float):  Weight of the loss. Defaults to 1.0.
        class_weight (List[float], optional): The weight for each class with
            shape (C), C is the number of classes. Default None.
        pos_weight (List[float], optional): The positive weight for each
            class with shape (C), C is the number of classes. Only enabled in
            BCE loss when ``use_sigmoid`` is True. Default None.
    """

    def __init__(self,
                 reduction='mean',
                 loss_weight=None,
                 img_size=(128, 256),
                 learn_log_variance: Dict= dict(flag=False, requires_grad=True, channels=69, logvar_init=0.),
                 normalized_weight_cfg: Dict = dict(flag=False, queue_length=1000., channels=69, norm_value=1.),
                 class_weight: List=None,
                 pos_weight: List = None):
        super(LatWeighted_L2_LOSSv2, self).__init__()
        self.learn_log_variance =  learn_log_variance.get('flag')
        self.normlized_weight = normalized_weight_cfg.get('flag')
        assert not (
                self.learn_log_variance and self.normlized_weight
        ), 'learn_log_variance and normlized_weight could not be set simultaneously'

        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.pos_weight = pos_weight
        self.normalized_weight_cfg = normalized_weight_cfg
        self.logvar = None

        weights = weights_torch_channels(img_size)
        self.lat_weights = nn.Parameter(weights, requires_grad=True)
        
        if self.learn_log_variance:
            self.channels = learn_log_variance.get('channels')
            self.logvar = nn.Parameter(
                torch.ones(size=(1,self.channels,1,1),requires_grad=learn_log_variance.get('requires_grad')) * learn_log_variance.get('logvar_init')
                )
        
        self.count = 0

    def forward(self,
                pred,
                label,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        # only BCE loss has pos_weight
        if self.pos_weight is not None and self.use_sigmoid:
            pos_weight = pred.new_tensor(self.pos_weight)
            kwargs.update({'pos_weight': pos_weight})
        else:
            pos_weight = None

        loss = torch.square(pred-label)

        assert self.logvar.data.ndim == loss.ndim
        loss = loss / (torch.exp(self.logvar)) + self.logvar
        loss = self.lat_weights * loss

        loss = loss.mean(dim=(-1,-2))

        if get_rank() == 0:
            self.count+=1
            if self.count%100==0:
                print_log(f'loss channel weight:{self.logvar.data.squeeze()}', logger='current')
        return loss.mean()


if __name__ == '__main__':
    learn_log_variance=dict(flag=True, channels=65, logvar_init=0., requires_grad=True)

    loss_function = LatWeighted_L2_LOSSv2(img_size=(128, 256), learn_log_variance=learn_log_variance)

    prediction = torch.randn([1, 65, 128, 256])
    labels = torch.ones([1, 65, 128, 256])

    loss = loss_function(prediction, labels)

    print(loss)