import yaml
import argparse

def _load_hparams(filename):
    if filename:
        with open(filename) as fin:
            return yaml.load(fin, Loader=yaml.Loader)
    return {}


class HParams(object):
    def __init__(self, filepath=None, args=None, set_hparams=None):
        super(HParams, self).__setattr__('hp_loaded', _load_hparams(filepath))
        super(HParams, self).__setattr__('hp_defined', vars(args) if args else {})
        super(HParams, self).__setattr__('hp_runtime', {})
        if set_hparams:
            set_hparams(self)

    def __getattr__(self, item):
        if item in self.hp_runtime:
            return self.hp_runtime[item]
        if item in self.hp_defined:
            return self.hp_defined[item]
        if item in self.hp_loaded:
            return self.hp_loaded[item]
        raise AttributeError

    def __setattr__(self, key, value):
        self.hp_runtime[key] = value

    def get_dict(self):
        hp = {k: v for k, v in self.hp_runtime.items()}
        hp.update({k: v for k, v in self.hp_defined.items() if k not in hp})
        hp.update({k: v for k, v in self.hp_loaded.items() if k not in hp})
        return hp

    def get_namesapce(self):
        return argparse.Namespace(**self.get_dict())