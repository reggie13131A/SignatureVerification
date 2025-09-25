import math
import warnings
from typing import List
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, RMSprop, AdamW
from torch.optim.optimizer import Optimizer, required
import re 
from torch.optim.lr_scheduler import _LRScheduler
from Network_modules import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(1)


class Triplet_Model(nn.Module):
    def __init__(self, args):
        super(Triplet_Model, self).__init__()
        self.args = args
        
        self.encoder = InceptionResNet_V2(pretrained=False)
        # self.encoder.load_state_dict(torch.load(args.load_model), strict=True)
        # 这里没有加BN层，是导致神经元坏死的原因
        self.projector = nn.Sequential(nn.Linear(1536, 1536), nn.ReLU(), nn.Linear(1536, 1536))
        self.enc_params = self.encoder.parameters()
        self.proj_params = self.projector.parameters() 

        # 这个是用来计算损失的，怎么计算的？太奇怪了
        self.criterion = nn.TripletMarginLoss(margin=args.margin)
        self.optimizer = AdamW([
            {"params" : self.enc_params,  "lr" : self.args.learning_rate_AE},
            {"params" : self.proj_params, "lr" : self.args.learning_rate},
        ], self.args.learning_rate, weight_decay=0.0005)
        # self.optimizer = RMSprop([
        #     {"params" : self.enc_params,  "lr" : self.args.learning_rate_AE},
        #     {"params" : self.proj_params, "lr" : self.args.learning_rate},
        # ], self.args.learning_rate, weight_decay=0.0005)
        # self.scheduler = LinearWarmupCosineAnnealingLR(self.optimizer, self.args.warmup_epochs, self.args.max_epochs)

    def __check_tensor__(self, tensor, name):
        """递归检查张量或嵌套结构中的张量"""
        if isinstance(tensor, torch.Tensor):
            if torch.isnan(tensor).any():
                print(f"⚠️ 输入数据 {name} 包含 NaN")
                raise RuntimeError(f"输入数据 {name} 包含 NaN，终止训练")
            if torch.isinf(tensor).any():
                print(f"⚠️ 输入数据 {name} 包含 Inf")
                raise RuntimeError(f"输入数据 {name} 包含 Inf，终止训练")
        elif isinstance(tensor, (list, tuple)):
            for i, t in enumerate(tensor):
                self.__check_tensor__(t, f"{name}[{i}]")
        # 跳过非张量字段（如字符串、整数等），不再报错


    
    def train_model(self, batch):
        self.train()
        self.optimizer.zero_grad()
        intra_triplet_loss, inter_triplet_loss = 0.0, 0.0

        # anchor是查询集
        A_feat = self.projector(self.encoder(batch['anchor'].to(device), pool=True))
        # positive是真的
        P_feat = self.projector(self.encoder(batch['positive'].to(device), pool=True))
        # 伪造的签名
        N1_feat = self.projector(self.encoder(batch['negative_intra'].to(device), pool=True))
        # 别人的签名（不属于查询的）
        N2_feat = self.projector(self.encoder(batch['negative_inter'].to(device), pool=True))
        
        intra_triplet_loss += self.criterion(A_feat, P_feat, N1_feat)
        inter_triplet_loss += self.criterion(A_feat, P_feat, N2_feat)
        print(f'Intra-class Triplet Loss: {intra_triplet_loss.item():.4f} | Inter-class Triplet Loss: {inter_triplet_loss.item():.4f}')

        (intra_triplet_loss + self.args.lambd * inter_triplet_loss).backward()
        # 检查输入数据是否异常
        for key, value in batch.items():
            self.__check_tensor__(value, key)

        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"\n当前学习率: {current_lr:.6f}")
        total_grad = sum(p.grad.abs().sum() for p in self.parameters() if p.grad is not None)
        print(f"梯度总量: {total_grad.item():.6f}")
        # 裁剪
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        print("backward caculation完成")

        return intra_triplet_loss.item(), inter_triplet_loss.item()


### optimizer.py ###
class LARS(Optimizer):
    """
    Layer-wise Adaptive Rate Scaling for large batch training.
    Introduced by "Large Batch Training of Convolutional Networks" by Y. You,
    I. Gitman, and B. Ginsburg. (https://arxiv.org/abs/1708.03888)
    """

    def __init__(
        self,
        params,
        lr=required,
        momentum=0.9,
        use_nesterov=False,
        weight_decay=0.0,
        exclude_from_weight_decay=None,
        exclude_from_layer_adaptation=None,
        classic_momentum=True,
        eeta=0.001,
    ):
        """Constructs a LARSOptimizer.
        Args:
        lr: A `float` for learning rate.
        momentum: A `float` for momentum.
        use_nesterov: A 'Boolean' for whether to use nesterov momentum.
        weight_decay: A `float` for weight decay.
        exclude_from_weight_decay: A list of `string` for variable screening, if
            any of the string appears in a variable's name, the variable will be
            excluded for computing weight decay. For example, one could specify
            the list like ['batch_normalization', 'bias'] to exclude BN and bias
            from weight decay.
        exclude_from_layer_adaptation: Similar to exclude_from_weight_decay, but
            for layer adaptation. If it is None, it will be defaulted the same as
            exclude_from_weight_decay.
        classic_momentum: A `boolean` for whether to use classic (or popular)
            momentum. The learning rate is applied during momeuntum update in
            classic momentum, but after momentum for popular momentum.
        eeta: A `float` for scaling of learning rate when computing trust ratio.
        name: The name for the scope.
        """

        self.epoch = 0
        defaults = dict(
            lr=lr,
            momentum=momentum,
            use_nesterov=use_nesterov,
            weight_decay=weight_decay,
            exclude_from_weight_decay=exclude_from_weight_decay,
            exclude_from_layer_adaptation=exclude_from_layer_adaptation,
            classic_momentum=classic_momentum,
            eeta=eeta,
        )

        super(LARS, self).__init__(params, defaults)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.use_nesterov = use_nesterov
        self.classic_momentum = classic_momentum
        self.eeta = eeta
        self.exclude_from_weight_decay = exclude_from_weight_decay
        # exclude_from_layer_adaptation is set to exclude_from_weight_decay if the
        # arg is None.
        if exclude_from_layer_adaptation:
            self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
        else:
            self.exclude_from_layer_adaptation = exclude_from_weight_decay

    def step(self, epoch=None, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        if epoch is None:
            epoch = self.epoch
            self.epoch += 1

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            eeta = group["eeta"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                param = p.data
                grad = p.grad.data

                param_state = self.state[p]

                # TODO: get param names
                # if self._use_weight_decay(param_name):
                grad += self.weight_decay * param

                if self.classic_momentum:
                    trust_ratio = 1.0

                    # TODO: get param names
                    # if self._do_layer_adaptation(param_name):
                    w_norm = torch.norm(param)
                    g_norm = torch.norm(grad)

                    device = g_norm.get_device()
                    trust_ratio = torch.where(
                        w_norm.gt(0),
                        torch.where(
                            g_norm.gt(0),
                            (self.eeta * w_norm / g_norm),
                            torch.Tensor([1.0]).to(device),
                        ),
                        torch.Tensor([1.0]).to(device),
                    ).item()

                    scaled_lr = lr * trust_ratio
                    if "momentum_buffer" not in param_state:
                        next_v = param_state["momentum_buffer"] = torch.zeros_like(
                            p.data
                        )
                    else:
                        next_v = param_state["momentum_buffer"]

                    next_v.mul_(momentum).add_(scaled_lr, grad)
                    if self.use_nesterov:
                        update = (self.momentum * next_v) + (scaled_lr * grad)
                    else:
                        update = next_v

                    p.data.add_(-update)
                else:
                    raise NotImplementedError

        return loss

    def _use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _do_layer_adaptation(self, param_name):
        """Whether to do layer-wise learning rate adaptation for `param_name`."""
        if self.exclude_from_layer_adaptation:
            for r in self.exclude_from_layer_adaptation:
                if re.search(r, param_name) is not None:
                    return False
        return True


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """
    Sets the learning rate of each parameter group to follow a linear warmup schedule
    between warmup_start_lr and base_lr followed by a cosine annealing schedule between
    base_lr and eta_min.
    .. warning::
        It is recommended to call :func:`.step()` for :class:`LinearWarmupCosineAnnealingLR`
        after each iteration as calling it after each epoch will keep the starting lr at
        warmup_start_lr for the first epoch which is 0 in most cases.
    .. warning::
        passing epoch to :func:`.step()` is being deprecated and comes with an EPOCH_DEPRECATION_WARNING.
        It calls the :func:`_get_closed_form_lr()` method for this scheduler instead of
        :func:`get_lr()`. Though this does not change the behavior of the scheduler, when passing
        epoch param to :func:`.step()`, the user should call the :func:`.step()` function before calling
        train and validation methods.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
        verbose = True
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Compute learning rate using chainable form of the scheduler
        """
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        elif self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        elif (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return [
                group["lr"] + (base_lr - self.eta_min) *
                (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs))) /
            (
                1 +
                math.cos(math.pi * (self.last_epoch - self.warmup_epochs - 1) / (self.max_epochs - self.warmup_epochs))
            ) * (group["lr"] - self.eta_min) + self.eta_min for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """
        Called when epoch is passed as a param to the `step` function of the scheduler.
        """
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr + self.last_epoch * (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min + 0.5 * (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            for base_lr in self.base_lrs
        ]
