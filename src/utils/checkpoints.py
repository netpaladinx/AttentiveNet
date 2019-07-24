import os
import shutil
import torch


def load_checkpoint(checkpoint_path, cur_gpu, model, optimizer, lr_scheduler):
    checkpoint = torch.load(checkpoint_path,
                            map_location=lambda storage, loc: storage.cuda(cur_gpu) if cur_gpu >= 0 else storage)
    start_epoch = checkpoint['epoch'] + 1
    best_acc1 = checkpoint['best_acc1']
    best_acc5 = checkpoint['best_acc5']
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    return start_epoch, model, optimizer, lr_scheduler, best_acc1, best_acc5


def save_checkpoint(model_dir, epoch, model, optimizer, lr_scheduler, best_acc1, best_acc5, is_best):
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'best_acc1': best_acc1,
        'best_acc5': best_acc5
    }
    filename = 'checkpoint-epoch%d.pt' % (epoch + 1)
    torch.save(checkpoint, os.path.join(model_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(model_dir, filename), os.path.join(model_dir, 'checkpoint-best.pt'))
