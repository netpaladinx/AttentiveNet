import torch
import torch.distributed as dist


def batch_reduce(bs, *tensors, dtype=torch.float):
    bs = bs.detach().clone()
    tensors = [t.detach().clone().float().mul_(bs) for t in tensors]

    bs_handle = dist.all_reduce(bs, op=dist.ReduceOp.SUM, async_op=True)
    handles = []
    for t in tensors:
        handles.append(dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True))

    bs_handle.wait()
    for h in handles:
        h.wait()

    tensors = [t.div_(bs).to(dtype) for t in tensors]
    return [bs] + tensors