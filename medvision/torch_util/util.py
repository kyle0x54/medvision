from enum import Enum, unique, auto
import numpy as np
import torch


@unique
class ModeKey(Enum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()


def nograd(f):
    def decorator(*args, **kwargs):
        with torch.no_grad():
            return f(*args, **kwargs)
    return decorator


def make_np(x):
    if isinstance(x, np.ndarray):
        return x

    if 'torch' in str(type(x)):
        return x.cpu().numpy()

    raise NotImplementedError(
        'Got {}, but expected numpy array or torch tensor.'.format(type(x)))


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
