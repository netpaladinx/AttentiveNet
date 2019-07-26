import torch
import torch.nn as nn
import torch.nn.functional as F

class Lambda(nn.Module):
    def __init__(self, fn):
        super(Lambda, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


def cross_entropy(input, target, label_smoothing=0.0):
    ''' input (logits): N X C
        target: N (int64)
    '''
    if label_smoothing > 0.0:
        input = F.log_softmax(input, dim=1)
        n_classes = input.size(1)
        target = F.one_hot(target, num_classes=n_classes)
        target = (1 - label_smoothing) * target.float() + label_smoothing / n_classes
        return F.kl_div(input, target, reduction='batchmean')
    else:
        return F.cross_entropy(input, target)


def convert_to_half(model):
    model = model.half()
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.float()
    return model


def get_parameters(model, include=None, exclude=None, clone=False):
    params, params_clone = [], []
    for m in model.modules():
        if include and not isinstance(m, include):
            continue
        if exclude and isinstance(m, exclude):
            continue
        for p in m.parameters(recurse=False):
            params.append(p)
            if clone:
                params_clone.append(p.clone().type(torch.cuda.FloatTensor).detach().requires_grad_())
    return params, params_clone


def copy_grads(params_src, params_dst):
    for p_src, p_dst in zip(params_src, params_dst):
        p_dst.grad = torch.zeros_like(p_dst.data)
        if p_src.grad is not None:
            p_dst.grad.data.copy_(p_src.grad.data)


def copy_params(params_src, params_dst):
    for p_src, p_dst in zip(params_src, params_dst):
        p_dst.data.copy_(p_src.data)
