import os
import shutil
import cpuinfo
import psutil
import multiprocessing

import torch


def gather_run_info():
    torch_info = {
        "torch_version": torch.__version__,
    }

    info = cpuinfo.get_cpu_info()
    cpu_info = {
        "n_cores": multiprocessing.cpu_count(),
        "cpu_info": info['brand'],
        "mhz_per_cpu": info['hz_advertised_raw'][0] / 1.0e6
    }

    n_devices = torch.cuda.device_count()
    gpu_info = {
        "n_gpus": n_devices,
        "cuda_available": torch.cuda.is_available(),
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''
    }

    vmem = psutil.virtual_memory()
    mem_info = {
        "mem_total": '%.1f GB' % round(vmem.total / 1024**3, 1),
        "mem_available": '%.1f GB' % round(vmem.available / 1024**3, 1)
    }

    run_info = {
        "torch_info": torch_info,
        "cpu_info": cpu_info,
        "gpu_info": gpu_info,
        "mem_info": mem_info
    }
    return run_info


def clean_dirs(dirs):
    if not isinstance(dirs, (tuple, list)):
        dirs = [dirs]
    for d in dirs:
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)
