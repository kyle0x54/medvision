from collections import OrderedDict
import torch
import medvision as mv


def _weights_to_cpu(state_dict_gpu):
    """ Copy a model state_dict to cpu.

    Args:
        state_dict_gpu (OrderedDict): model weights on GPU.

    Return:
        (OrderedDict): model weights on CPU.
    """
    state_dict_cpu = OrderedDict()
    for key in state_dict_gpu:
        state_dict_cpu[key] = state_dict_gpu[key].cpu()
    return state_dict_cpu


def save_checkpoint(model, path, optimizer=None, metadata=None):
    """ Save checkpoint to file.

    The checkpoint will have 3 fields: ``metadata``, ``state_dict`` and
    ``optimizer``.

    Args:
        model (Module): module whose params are to be saved.
        path (str): path to save the checkpoint file.
        optimizer ('Optimizer', optional): optimizer to be saved.
        metadata (dict, optional): metadata to be saved in checkpoint.
    """
    assert isinstance(metadata, (dict, type(None)))
    if metadata is None:
        metadata = {}

    mv.mkdirs(mv.parentdir(path))

    # if wrapped by nn.DataParallel, remove the wrapper
    if hasattr(model, 'module'):
        model = model.module

    # make a checkpoint
    checkpoint = {'state_dict': _weights_to_cpu(model.state_dict())}
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    if metadata is not None:
        checkpoint['metadata'] = metadata

    torch.save(checkpoint, path)


def load_checkpoint(model, path, map_location='cpu', strict=True):
    """ Load checkpoint from a file.

    Args:
        model (Module): module to load checkpoint.
        path (str): checkpoint weights file path.
        map_location (str): same as :func:`torch.load`.
        strict (bool): whether to allow different params for the model and
            checkpoint.

    TODO: support load checkpoint from url with torch.utils.model_zoo
    """
    # load checkpoint from file
    checkpoint = torch.load(path, map_location=map_location)

    # get state_dict from checkpoint
    if isinstance(checkpoint, OrderedDict):
        # checkpoint saved by torch.save(model.state_dict(), path)
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        # checkpoint saved by save_checkpoint()
        state_dict = checkpoint['state_dict']
    else:
        raise RuntimeError(
            'No state_dict found in checkpoint file {}'.format(path))

    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    # load state_dict
    if hasattr(model, 'module'):
        model.module.load_state_dict(state_dict, strict)
    else:
        model.load_state_dict(state_dict, strict)


def save_ckpt_to_dir(model, ckpt_dir, identifier):
    """ Save checkpoint to file and make a soft link to the latest ckpt.

    Args:
        model (Module): module whose params are to be saved.
        ckpt_dir (str): directory to save the checkpoint file.
        identifier (str): ckpt identifier (e.g. number of epochs).
    """
    model_path = mv.joinpath(ckpt_dir, str(identifier) + '.pth')
    mv.save_checkpoint(model, model_path)

    link_path = mv.joinpath(ckpt_dir, 'latest' + '.pth')
    mv.symlink(mv.abspath(model_path), link_path)


def load_ckpt_from_dir(model, ckpt_dir, identifier='latest'):
    """ Load checkpoint from a directory with given identifier.

    Args:
        model (Module): module whose params are to be saved.
        ckpt_dir (str): directory to load the checkpoint file.
        identifier (str): ckpt identifier (e.g. number of epochs).
    """
    model_path = mv.joinpath(ckpt_dir, str(identifier) + '.pth')
    mv.load_checkpoint(model, model_path)
