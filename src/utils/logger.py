import os
import contextlib
import threading
import datetime
import json
import yaml
from collections import OrderedDict

from utils import gather_run_info


_logger = None
_logger_lock = threading.Lock()


def get_logger():
    return _logger


def _set_logger(log_dir):
    _logger_lock.acquire()
    try:
        global _logger
        _logger = Logger(log_dir)
    finally:
        _logger_lock.release()
    return _logger


@contextlib.contextmanager
def logger_context(log_dir, rank=0):
    (rank == -1 or rank == 0) and _set_logger(log_dir)
    try:
        yield
        _logger and _logger.on_finish('success')
    except Exception:
        _logger and _logger.on_finish('failure')
        raise


class Logger(object):
    def __init__(self, log_dir):
        self.log_dir = log_dir
        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)
        self.metric_file = 'metric.log'
        self.summary_file = 'summary.log'
        self.runinfo_file = 'runinfo.log'
        self.hparams_file = 'hparams.log'
        self.metric_file_handler = open(os.path.join(self.log_dir, self.metric_file), 'w')
        self.summary_file_handler = open(os.path.join(self.log_dir, self.summary_file), 'w')

    def log_metrics(self, metrics, epoch, step, mode):
        ''' metrics: [(name, value), ...]
            mode: 'train' or 'val'
        '''
        mtr = OrderedDict(
            [('mode', mode), ('epoch', epoch), ('step', step)] +
            [(name, float(value)) for name, value in metrics] +
            [('timestamp', datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ"))]
        )
        json.dump(mtr, self.metric_file_handler)
        self.metric_file_handler.write('\n')
        self.metric_file_handler.flush()

    def log_summaries(self, summaries, epoch, step, mode):
        ''' summaries: [(name, value), ...]
            mode: 'train' or 'val'
        '''
        smr = OrderedDict(
            [('mode', mode), ('epoch', epoch), ('step', step)] +
            [(name, float(value)) for name, value in summaries] +
            [('timestamp', datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ"))]
        )
        json.dump(smr, self.summary_file_handler)
        self.summary_file_handler.write('\n')
        self.summary_file_handler.flush()

    def log_run_info(self):
        info = gather_run_info()
        with open(os.path.join(self.log_dir, self.runinfo_file), 'w') as fout:
            yaml.dump(info, fout)
            fout.write('\n')

    def log_hparams(self, hparams):
        hp = vars(hparams)
        with open(os.path.join(self.log_dir, self.hparams_file), 'w') as fout:
            yaml.dump(hp, fout)
            fout.write('\n')

    def on_finish(self, status):
        self.metric_file_handler.flush()
        self.metric_file_handler.close()

        self.summary_file_handler.flush()
        self.summary_file_handler.close()
