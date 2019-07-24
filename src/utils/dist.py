import torch.distributed as dist


def batch_reduce(bs, *tensors):
    tensors = [t.clone().detach().mul_(bs) for t in tensors]

    dist.all_reduce(bs, op=dist.ReduceOp.SUM)
    for t in tensors:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)

    tensors = [t.div_(bs) for t in tensors]
    return [bs] + tensors