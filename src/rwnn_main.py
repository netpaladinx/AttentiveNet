import os
import argparse
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist

import rwnn_models as models
from imagenet_dataset import get_train_loader, get_val_loader
from utils import clean_dirs
from utils.hyperparameters import HParams
from utils.nn import cross_entropy, convert_to_half, get_parameters, copy_grads, copy_params
from utils.lr_schedulers import scale_initial_learning_rate, get_lr_scheduler
from utils.dist import batch_reduce
from utils.metrics import accuracy
from utils.meters import AverageMeter
from utils.logger import logger_context, get_logger
from utils.monitors import get_progress_monitor
from utils.checkpoints import load_checkpoint, save_checkpoint


MODEL_NAME = 'rwnn_small_ws'
EXPERIMENT_ID = '{}_{}'.format(MODEL_NAME, datetime.datetime.utcnow().strftime("%Y-%m-%d"))


def define_args():
    parser = argparse.ArgumentParser(description='rwnn_main')
    parser.add_argument('--model_name', default=MODEL_NAME, type=str, help='model_name')
    parser.add_argument('--experiment_id', default=EXPERIMENT_ID, type=str, help='experiment_id')
    parser.add_argument('--config_file', default='../config/{}_hparams.yaml'.format(MODEL_NAME), type=str, help='config_file')
    parser.add_argument('--checkpoint', default='', type=str, help='checkpoint')
    parser.add_argument('--local_debug', default=False, action='store_true', help='local_debug')
    parser.add_argument('--save_model', default=False, action='store_true', help='save_model')
    parser.add_argument('--lr_scheduler', default='cosine_annealing_warm_restarts', type=str, help='lr_scheduler')
    return parser.parse_args()


def set_hparams(hparams):
    hparams.model_dir = os.path.join(hparams.model_dir, hparams.experiment_id)
    hparams.log_dir = os.path.join(hparams.log_dir, hparams.experiment_id)

    if torch.cuda.is_available():
        hparams.world_size = torch.cuda.device_count()
        if hparams.world_size > 1:
            hparams.distributed_mode = 'gpus'
            hparams.n_data_loading_workers = (hparams.n_data_loading_workers +
                                              hparams.world_size - 1) // hparams.world_size
        else:
            hparams.distributed_mode = 'gpu'
    else:
        hparams.world_size = 1
        hparams.distributed_mode = 'cpu'
        hparams.fp16 = False

    hparams.global_batch_size = hparams.per_replica_batch_size * hparams.world_size
    hparams.steps_per_epoch = (hparams.n_train_images + hparams.global_batch_size - 1) // hparams.global_batch_size
    hparams.initial_learning_rate = scale_initial_learning_rate(hparams.initial_learning_rate, hparams.global_batch_size)
    hparams.loss_scale = 128 if hparams.fp16 else 1

    if hparams.lr_scheduler == 'multi_step_lr_with_warmup':
        hparams.lr_milestones = [30, 60, 80, 90]
        hparams.lr_decay_rate = 0.1
        hparams.lr_warmup_epochs = 5
        hparams.lr_factor_min = 0.0001
        hparams.epochs = 100
    elif hparams.lr_scheduler == 'cosine_annealing_warm_restarts':
        hparams.lr_T_0 = 4
        hparams.lr_T_mult = 2
        hparams.lr_factor_min = 0.0001
        hparams.epochs = 252  # 4 * (1 + 2 + 4 + 8 + 16 + 32) = 4 * 63

    if hparams.local_debug:
        hparams.per_replica_batch_size = 64
        hparams.print_freq = 1
        hparams.n_classes = 12
        hparams.n_train_images = 14300
        hparams.n_valid_images = 172

def run(cur_gpu, hparams):
    if hparams.distributed_mode == 'gpus':
        dist.init_process_group(backend=hparams.dist_backend, init_method=hparams.dist_url,
                                world_size=hparams.world_size, rank=cur_gpu)


    model = getattr(models, hparams.model_name)(hparams.n_classes, hparams.n_channels)

    if cur_gpu >= 0:
        torch.cuda.set_device(cur_gpu)
        model.cuda()

    if hparams.fp16:
        model = convert_to_half(model)

    if hparams.distributed_mode == 'gpus':
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cur_gpu], output_device=cur_gpu)

    criterion = cross_entropy

    params_no_bn, params_no_bn_clone = get_parameters(model, exclude=(nn.BatchNorm2d, nn.SyncBatchNorm, nn.GroupNorm),
                                                      clone=hparams.fp16)
    params_bn, params_bn_clone = get_parameters(model, include=(nn.BatchNorm2d, nn.SyncBatchNorm, nn.GroupNorm),
                                                clone=hparams.fp16)
    optimizer = optim.SGD([
        {'params': params_no_bn_clone if hparams.fp16 else params_no_bn, 'weight_decay': hparams.weight_decay},
        {'params': params_bn_clone if hparams.fp16 else params_bn, 'weight_decay': 0.0}
    ], lr=hparams.initial_learning_rate, momentum=hparams.momentum)

    lr_scheduler = get_lr_scheduler(hparams.lr_scheduler, optimizer, hparams)

    best_acc1 = 0
    best_acc5 = 0
    start_epoch = hparams.start_epoch
    if hparams.checkpoint and os.path.isfile(hparams.checkpoint):
        start_epoch, model, optimizer, lr_scheduler, best_acc1, best_acc5 = load_checkpoint(
            hparams.checkpoint, cur_gpu, model, optimizer, lr_scheduler)

    torch.backends.cudnn.benchmark = True

    train_loader, train_sampler = get_train_loader(hparams.data_dir, hparams.image_size,
                                                   hparams.per_replica_batch_size,
                                                   hparams.n_data_loading_workers,
                                                   hparams.distributed_mode,
                                                   hparams.world_size, cur_gpu)
    val_loader = get_val_loader(hparams.data_dir, hparams.image_size, hparams.per_replica_batch_size,
                                hparams.n_data_loading_workers, hparams.distributed_mode,
                                hparams.world_size, cur_gpu)

    if hparams.evaluate:
        return validate(cur_gpu, val_loader, model, criterion, 0, hparams)

    monitor = get_progress_monitor(cur_gpu, hparams.log_dir,
                                   hparams.steps_per_epoch, hparams.epochs,
                                   hparams.print_freq, start_epoch)

    for epoch in range(start_epoch, hparams.epochs):
        monitor and monitor.before_epoch()

        if train_sampler:
            train_sampler.set_epoch(epoch)
        train(cur_gpu, train_loader, model, criterion, optimizer, lr_scheduler,
              params_no_bn + params_bn, params_no_bn_clone + params_bn_clone,
              epoch, hparams, monitor)

        loss, acc1, acc5 = validate(cur_gpu, val_loader, model, criterion, epoch, hparams)

        monitor and monitor.after_epoch(loss, acc1, acc5)

        if hparams.save_model and cur_gpu in (-1, 0):
            is_best = acc1 > best_acc1
            best_acc1 = acc1 if is_best else best_acc1
            save_checkpoint(hparams.model_dir, epoch, model, optimizer, lr_scheduler,
                            best_acc1, best_acc5, is_best)

    if hparams.distributed_mode == 'gpus':
        dist.destroy_process_group()

    monitor and monitor.end()


def train(cur_gpu, train_loader, model, criterion, optimizer, lr_scheduler,
          params, params_clone, epoch, hparams, monitor):
    model.train()

    loss_meter = AverageMeter('train_loss')
    acc1_meter = AverageMeter('train_acc1')
    acc5_meter = AverageMeter('train_acc5')

    for i, (image, target) in enumerate(train_loader):
        monitor and monitor.before_step()

        if cur_gpu >= 0:
            image = image.cuda(cur_gpu, non_blocking=True)
            target = target.cuda(cur_gpu, non_blocking=True)

        if hparams.fp16:
            image = image.half()

        output = model(image)

        if hparams.fp16:
            output = output.float()

        loss = criterion(output, target, label_smoothing=hparams.label_smoothing)
        loss_ = loss.data.clone()
        acc1, acc5 = accuracy(output, target, topk=(1,5))
        bs = torch.tensor(image.size(0), device='cuda:%d' % cur_gpu if cur_gpu >= 0 else None)

        if hparams.distributed_mode == 'gpus':
            bs, loss_, acc1, acc5 = batch_reduce(bs, loss_, acc1, acc5)

        loss_meter.update(loss.item(), bs.item())
        acc1_meter.update(acc1.item(), bs.item())
        acc5_meter.update(acc5.item(), bs.item())

        if hparams.fp16:
            loss = loss * hparams.loss_scale
            model.zero_grad()
            loss.backward()
            copy_grads(params, params_clone)
            for p in params_clone:
                p.grad.data.div_(hparams.loss_scale)
            optimizer.step()
            copy_params(params_clone, params)
            torch.cuda.synchronize()
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        step = i + 1
        lr_scheduler.step(epoch=epoch + step / hparams.steps_per_epoch)

        if step % hparams.print_freq == 0:
            metrics = [('train_loss', loss_meter.result),
                       ('train_acc1', acc1_meter.result),
                       ('train_acc5', acc5_meter.result),
                       ('lr', optimizer.param_groups[0]['lr'])]

            logger = get_logger()
            if logger:
                # epoch: zero-indexed for code, one-indexed for human reader
                logger.log_metrics(metrics, epoch + 1, step, 'train')

        monitor and monitor.after_step(str(loss_meter))

    metrics = [('train_loss', loss_meter.result),
               ('train_acc1', acc1_meter.result),
               ('train_acc5', acc5_meter.result),
               ('lr', optimizer.param_groups[0]['lr'])]

    logger = get_logger()
    if logger:
        logger.log_metrics(metrics, epoch + 1, step, 'train')


def validate(cur_gpu, val_loader, model, criterion, epoch, hparams):
    model.eval()

    loss_meter = AverageMeter('val_loss')
    acc1_meter = AverageMeter('val_acc1')
    acc5_meter = AverageMeter('val_acc5')

    for i, (image, target) in enumerate(val_loader):
        with torch.no_grad():
            if cur_gpu >= 0:
                image = image.cuda(cur_gpu, non_blocking=True)
                target = target.cuda(cur_gpu, non_blocking=True)

            if hparams.fp16:
                image = image.half()

            output = model(image)

            if hparams.fp16:
                output = output.float()

            loss = criterion(output, target, label_smoothing=hparams.label_smoothing)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            bs = torch.tensor(image.size(0), device='cuda:%d' % cur_gpu if cur_gpu >= 0 else None)

            if hparams.distributed_mode == 'gpus':
                bs, loss, acc1, acc5 = batch_reduce(bs, loss, acc1, acc5)

            loss_meter.update(loss.item(), bs.item())
            acc1_meter.update(acc1.item(), bs.item())
            acc5_meter.update(acc5.item(), bs.item())

    metrics = [('val_loss', loss_meter.result),
               ('val_acc1', acc1_meter.result),
               ('val_acc5', acc5_meter.result)]

    logger = get_logger()
    if logger:
        logger.log_metrics(metrics, epoch + 1, 0, 'val')

    return loss_meter.result, acc1_meter.result, acc5_meter.result


def main_worker(i, hparams):
    with logger_context(hparams.log_dir, rank=i):
        logger = get_logger()
        if logger:
            logger.log_run_info()
            logger.log_hparams(hparams)

        run(i, hparams)


def main():
    args = define_args()
    hparams = HParams(filepath=args.config_file, args=args, set_hparams=set_hparams).get_namesapce()

    clean_dirs([hparams.model_dir, hparams.log_dir])

    if hparams.distributed_mode == 'gpus':
        mp.spawn(main_worker, nprocs=hparams.world_size, args=(hparams,))
    else:
        cur_gpu = 0 if hparams.distributed_mode == 'gpu' else -1
        main_worker(cur_gpu, hparams)


if __name__ == '__main__':
    main()