import torch


def nograd(f):
    def decorator(*args, **kwargs):
        with torch.no_grad():
            return f(*args, **kwargs)
    return decorator
