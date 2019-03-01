from collections import OrderedDict
import random
from natsort import natsorted
import numpy as np
import medvision as mv


def make_dsmd(data):
    """ Make a dataset metadata.

    Args:
        data (dict): dataset metadata.
    """
    if isinstance(data, dict):
        dsmd = OrderedDict(natsorted(data.items()))
        return dsmd
    else:
        raise ValueError('dsmd only support dict type')


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


def match_dsmds(dt_src, gt_src, return_unmatched=False):
    """ Match 2 dataset metadata (e.g. annotation and detection result)

    Args:
        dt_src (dsmd): a dataset metadata instance.
        gt_src (dsmd): another dataset metadata instance.
        return_unmatched (bool): whether to return the unmatched
            part of dsmds.

    Return:
        (dsmd) matched dsmd of the first term.
        (dsmd) matched dsmd of the second term.
        (dsmd, optional) missing items in the 1st dsmd when
            matching with the 2nd dsmd.
        (dsmd, optional) unmatched items in the first dsmd when
            matching with the 2nd dsmd.
    """
    dt_keys = set(dt_src.keys())
    gt_keys = set(gt_src.keys())

    intersection_keys = dt_keys & gt_keys
    missing_keys = gt_keys - dt_keys
    unmatched_keys = dt_keys - gt_keys

    dt_dst = make_dsmd({key: dt_src[key] for key in intersection_keys})
    gt_dst = make_dsmd({key: gt_src[key] for key in intersection_keys})
    missing = make_dsmd({key: gt_src[key] for key in missing_keys})
    unmatched = make_dsmd({key: dt_src[key] for key in unmatched_keys})

    if return_unmatched:
        return dt_dst, gt_dst, missing, unmatched
    else:
        return dt_dst, gt_dst
