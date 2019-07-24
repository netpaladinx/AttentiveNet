import os
import warnings

import torch.utils.data as D
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from utils.data import DistributedSampler, DataPrefetcher

RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD = [0.229, 0.224, 0.225]

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


def get_train_loader(data_dir, image_size, per_replica_batch_size, n_data_loading_workers,
                     distributed_mode, world_size, cur_gpu):
    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'train'),
        transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=RGB_MEAN, std=RGB_STD)
        ]))
    if distributed_mode == 'gpus':
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=cur_gpu)
    else:
        train_sampler = None
    train_loader = D.DataLoader(train_dataset, batch_size=per_replica_batch_size, shuffle=(train_sampler is None),
                                num_workers=n_data_loading_workers, pin_memory=True, sampler=train_sampler)
    if cur_gpu >= 0:
        train_loader = DataPrefetcher(train_loader, cur_gpu)
    return train_loader, train_sampler


def get_val_loader(data_dir, image_size, per_replica_batch_size, n_data_loading_workers,
                   distributed_mode, world_size, cur_gpu):
    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'val'),
        transforms.Compose([
            transforms.Resize(int(image_size * 1.143)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=RGB_MEAN, std=RGB_STD)
        ]))
    if distributed_mode == 'gpus':
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=cur_gpu)
    else:
        val_sampler = None
    val_loader = D.DataLoader(val_dataset, batch_size=per_replica_batch_size, shuffle=False,
                              num_workers=n_data_loading_workers, pin_memory=True, sampler=val_sampler)
    if cur_gpu >= 0:
        val_loader = DataPrefetcher(val_loader, cur_gpu)
    return val_loader
