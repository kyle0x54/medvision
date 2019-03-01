import random
from natsort import natsorted
import numpy as np
import medvision as mv
from .classification import load_cls_dsmd, save_cls_dsmd
from .detection import load_det_dsmd, save_det_dsmd


def load_dsmd(dsmd_path, c2l_path=None, mode='cls'):
    """ Load dataset metadata.

    Args:
        dsmd_path (str): dataset metadata file path.
        c2l_path (str, optional): class-to-label file.
        mode (str): dataset mission, can be one of 'cls', 'seg', 'det'.

    Return:
        (OrderedDict): dataset metadata, refer to 'make_dsmd'.
    """
    if mode in ['cls', 'seg']:
        return load_cls_dsmd(dsmd_path)
    elif mode == 'det':
        return load_det_dsmd(dsmd_path, c2l_path)
    else:
        raise ValueError('only support cls, seg, det modes')


def save_dsmd(dsmd_path, data, c2l_path=None, auto_mkdirs=True, mode='cls'):
    """ Save dataset metadata to specified file.

    Args:
        dsmd_path (str): file path to save dataset metadata.
        data (dict): dataset metadata, refer to 'make_dsmd'.
        c2l_path (str, optional): class-to-label file.
        auto_mkdirs (bool): If the parent folder of `file_path` does
            not exist, whether to create it automatically.
        mode (str): dataset mission, can be one of 'cls', 'seg', 'det'.
    """
    if mode in ['cls', 'seg']:
        return save_cls_dsmd(dsmd_path, data, auto_mkdirs)
    elif mode == 'det':
        return save_det_dsmd(dsmd_path, data, c2l_path, auto_mkdirs)
    else:
        raise ValueError('only support cls, seg, det modes')


def load_c2l(c2l_path):
    """ Load class-to-label mapping.

    A class-to-label file defines the mapping from class_names to
    labels, which looks like (Note that the label value starts from 0)

    +------------------------------------------------------+
    | Class-to-Label File                                  |
    +------------------------------------------------------+
    |cat, 0                                                |
    |dog, 1                                                |
    |...                                                   |
    +------------------------------------------------------+

    Args:
        c2l_path (str): class-to-label file.

    Return:
        (OrderedDict): class-to-label mapping.
    """
    return load_dsmd(c2l_path)


def split_dsmd_file(dsmd_filepath, datasplit, shuffle=True, suffix='.csv'):
    """ Split a dataset medadata file into 3 parts.

    Split a dataset metadata file into 'train.csv', 'val.csv' and 'test.csv'.
    And put them in the same directory with specified dsmd file.

    dsmd_filepath (str): file path of dataset metadata.
    datasplit (dict[str, float]): how to split the dataset. e.g.
        {'train': 0.9, 'val': 0.1, 'test': 0.0}
    shuffle (bool): whether to shuffle the dataset before splitting.

    Note:
        0.0 < datasplit['train'] + datasplit['val'] + datasplit['test'] <= 1.0
        If there's no image in a split. The corresponding dsmd file will
        not be saved.
    """
    dsmd_dir = mv.parentdir(dsmd_filepath)

    dsmd = mv.load_dsmd(dsmd_filepath)
    num_total = len(dsmd)

    keys = list(dsmd.keys())
    if shuffle:
        random.shuffle(keys)

    sum_ratio = 0.0
    splits = {}
    for mode, ratio in datasplit.items():
        file_path = mv.joinpath(dsmd_dir, mode + suffix)
        splits[file_path] = int(num_total * ratio)
        sum_ratio += ratio
    assert 0.0 < sum_ratio <= 1.0

    start_index = 0
    for file_path, num_cur_split in splits.items():
        end_index = start_index + num_cur_split

        start_index = np.clip(start_index, 0, num_total)
        end_index = np.clip(end_index, 0, num_total)

        keys_split = keys[start_index:end_index]
        keys_split = natsorted(keys_split)
        dsmd_split = {keys: dsmd[keys] for keys in keys_split}
        if len(dsmd_split) != 0:
            mv.save_dsmd(file_path, dsmd_split)
            mv.save_dsmd(file_path, dsmd_split)

        start_index = end_index
