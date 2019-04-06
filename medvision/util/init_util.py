import logging.config
import os
import random
import numpy as np
import torch
import yaml
import medvision as mv


def init_logging(log_dir=None, config_file=None):
    if log_dir is None:
        log_dir = os.getcwd()

    if config_file is None:
        config_file = mv.joinpath(mv.parentdir(mv.parentdir(__file__)),
                                  'configs/default_log_config.yaml')

    with open(config_file, 'rt') as f:
        config = yaml.safe_load(f.read())
        config['handlers']['info_file_handler']['filename'] = \
            mv.joinpath(log_dir, 'info.log')
        config['handlers']['error_file_handler']['filename'] = \
            mv.joinpath(log_dir, 'error.log')
        mv.mkdirs(log_dir)
        logging.config.dictConfig(config)


def init_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)

    # code to handle reproducibility issue
    # https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_cuda_devices(cuda_devices=None):
    if cuda_devices is not None:
        cuda_devices = ','.join([str(d) for d in cuda_devices])
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_devices


def init_experiment(cfg):
    init_logging(cfg.LOGGING.LOG_DIR)
    init_random_seed(cfg.SYSTEM.RANDOM_SEED)
    init_cuda_devices(cfg.SYSTEM.CUDA_DEVICES)


init_system = init_experiment
