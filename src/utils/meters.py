
class AverageMeter(object):
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.cur = 0.0
        self.sum = 0.0
        self.cnt = 0

    def update(self, val, n=1):
        self.cur = val
        self.sum += val * n
        self.cnt += n

    def __str__(self):
        return '%s: %.3f (%.3f)' % (self.name, self.cur, self.sum / self.cnt)

    @property
    def result(self):
        return self.sum / self.cnt
