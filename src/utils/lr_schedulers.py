import math
from decimal import Decimal

import torch.optim as optim


def float_mod(a, b):
    return float(Decimal(str(a)) % Decimal(str(b)))


class CosineAnnealingWarmRestarts(optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, T_0, T_mult=1, factor_min=0, last_epoch=-1):
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.factor_min = factor_min
        self.T_cur = last_epoch
        super(CosineAnnealingWarmRestarts, self).__init__(optimizer, self.lr_lambda, last_epoch)

    def lr_lambda(self, epoch):
        if epoch > self.T_0:
            if self.T_mult == 1:
                self.T_cur = float(epoch, self.T_0)
            else:
                n = int(math.log(epoch / self.T_0 * (self.T_mult - 1) + 1, self.T_mult))
                self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                self.T_i = self.T_0 * (self.T_mult ** n)
        else:
            self.T_cur = epoch
        return self.factor_min + (1.0 - self.factor_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) * 0.5


def decayed_by_factor_every_n_epochs(optimizer, factor=0.1, epochs=30):
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: factor ** (epoch // epochs))


class MultiStepLRWithWarmup(optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, milestones, warmup_epochs=5, factor_min=0, gamma=0.1, last_epoch=-1):
        self.milestones = milestones
        self.warmup_epochs = warmup_epochs
        self.factor_min = factor_min
        self.gamma = gamma
        super(MultiStepLRWithWarmup, self).__init__(optimizer, self.lr_lambda, last_epoch)

    def lr_lambda(self, epoch):
        if epoch < self.warmup_epochs:
            return max(epoch / self.warmup_epochs, self.factor_min)
        else:
            decay = 1.0
            for i, m in enumerate(self.milestones):
                if epoch >= m:
                    decay *= self.gamma
                else:
                    break
            return decay


def get_lr_scheduler(name, optimizer, hparams):
    if name == 'multi_step_lr_with_warmup':
        return MultiStepLRWithWarmup(optimizer, hparams.lr_milestones, hparams.lr_warmup_epochs,
                                     factor_min=hparams.lr_factor_min, gamma=hparams.lr_decay_rate)
    if name == 'cosine_annealing_warm_restarts':
        return CosineAnnealingWarmRestarts(optimizer, hparams.lr_T_0, hparams.lr_T_mult, hparams.lr_factor_min)

def scale_initial_learning_rate(initial_learning_rate, global_batch_size, base_batch_size=256):
    return initial_learning_rate * global_batch_size / base_batch_size
