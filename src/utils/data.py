import math

import torch
import torch.utils.data as D
import torch.distributed as dist


class DistributedSampler(D.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, padding=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.padding = padding
        total_size = len(self.dataset)
        num_samples = int(math.ceil(total_size * 1.0 / num_replicas))
        if padding:
            self.num_samples = num_samples
            self.total_size = num_samples * self.num_replicas
        else:
            self.num_samples = \
                num_samples if total_size % num_replicas == 0 or total_size % num_replicas > rank \
                else num_samples - 1
            self.total_size = total_size

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()

        # add extra samples to make it evenly divisible
        if self.padding:
            indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class DataPrefetcher(object):
    def __init__(self, loader, cur_gpu, stop_after=None):
        self.loader = loader
        self.cur_gpu = cur_gpu
        self.dataset = loader.dataset
        self.stream = torch.cuda.Stream()
        self.stop_after = stop_after
        self.next_input = None
        self.next_target = None

    def __len__(self):
        return len(self.loader)

    def _preload(self, loader):
        try:
            self.next_input, self.next_target = next(loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(self.cur_gpu, non_blocking=True)
            self.next_target = self.next_target.cuda(self.cur_gpu, non_blocking=True)

    def __iter__(self):
        count = 0
        loader = iter(self.loader)
        self._preload(loader)
        while self.next_input is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            input = self.next_input
            target = self.next_target
            self._preload(loader)
            count += 1
            yield input, target
            if type(self.stop_after) is int and (count > self.stop_after):
                break
