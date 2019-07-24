import os
import time
import threading
import contextlib

_monitor = None
_monitor_lock = threading.Lock()


def get_monitor():
    return _monitor


def _set_monitor(log_dir, steps_per_epoch, epochs, print_freq, start_epoch):
    _monitor_lock.acquire()
    try:
        global _monitor
        _monitor = Monitor(log_dir, steps_per_epoch, epochs, print_freq, start_epoch)
    finally:
        _monitor_lock.release()
    return _monitor


@contextlib.contextmanager
def monitor_context(log_dir, steps_per_epoch, epochs, print_freq, start_epoch=0, rank=0):
    (rank == -1 or rank == 0) and _set_monitor(log_dir, steps_per_epoch, epochs, print_freq, start_epoch)
    try:
        yield
        _monitor and _monitor.on_finish('success')
    except Exception:
        _monitor and _monitor.on_finish('failure')
        raise


def time_str(t):
    h = int(t / 3600)
    t = int(t) % 3600
    m = int(t / 60)
    s = t % 60
    return '%02d:%02d:%02d' % (h, m, s)


class Monitor(object):
    def __init__(self, log_dir, steps_per_epoch, epochs, print_freq, start_epoch):
        self.log_dir = log_dir
        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)
        self.monitor_file_handler = open(os.path.join(self.log_dir, 'monitor.log'), 'w')

        self.print_freq = print_freq
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.start_epoch =start_epoch

        self.start_time = time.time()

        self.step = 0
        self.step_time_cost = 0
        self.step_count = 0

        self.epoch = start_epoch
        self.epoch_time_cost = 0
        self.epoch_count = 0

    def before_epoch(self):
        self.step_in_epoch = 0
        self.epoch_start_time = time.time()
        self.monitor_file_handler.write("===== EPOCH %d =====\n" % (self.epoch + 1))
        self.monitor_file_handler.flush()

    def before_step(self):
        self.step_start_time = time.time()

    def after_step(self, loss):
        self.step += 1
        self.step_in_epoch += 1
        self.step_time_cost += time.time() - self.step_start_time

        if self.step_in_epoch % self.print_freq == 0:
            time_past = time.time() - self.start_time
            secs_per_step = self.step_time_cost / self.step
            time_left_in_epoch = secs_per_step * (self.steps_per_epoch - self.step_in_epoch)
            time_left_in_total = secs_per_step * (self.steps_per_epoch * (self.epochs - self.start_epoch) - self.step)

            self.monitor_file_handler.write(
                "[Train - Step] [%s] [%s past] [%.2f secs per step] "
                "[Epoch %d: %d/%d steps, %s left] "
                "[Total: %d/%d steps, %s left]\n" %
                (loss, time_str(time_past),
                 secs_per_step, self.epoch + 1,
                 self.step_in_epoch,
                 self.steps_per_epoch,
                 time_str(time_left_in_epoch),
                 self.step,
                 self.steps_per_epoch * self.epochs,
                 time_str(time_left_in_total)))
            self.monitor_file_handler.flush()

    def after_epoch(self, loss, acc1, acc5):
        self.epoch += 1
        self.epoch_time_cost += time.time() - self.epoch_start_time

        time_past = time.time() - self.start_time
        secs_per_epoch = self.epoch_time_cost / (self.epoch - self.start_epoch)
        time_left = secs_per_epoch * (self.epochs - self.epoch)

        self.monitor_file_handler.write(
            "[Val - Epoch] [loss: %.4f, acc1: %.2f, acc5: %.2f], [%s past] [%.2f secs per epoch] "
            "[Total: %d/%d epochs, %s left]\n" %
            (loss, acc1, acc5,
             time_str(time_past), secs_per_epoch,
             self.epoch, self.epochs, time_str(time_left)))
        self.monitor_file_handler.flush()

    def on_finish(self, status):
        self.monitor_file_handler.flush()
        self.monitor_file_handler.close()
