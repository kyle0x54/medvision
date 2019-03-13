import numpy as np
import torch


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
