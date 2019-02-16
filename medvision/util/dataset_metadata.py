import random
from natsort import natsorted
import numpy as np
import medvision as mv


def load_dsmd(file_path):
    """ Load dataset metadata.

    Dataset metadata is a key-value pairs describing a dataset. For example,
    a dataset metadata looks like {'data/1.png': 1, 'data/2.png': 0, ...}.
    A dataset metadata file is a structured text file which looks like
    ---------------
    |data/1.png, 1|
    |data/2.png, 0|
    |...          |
    ---------------

    Args:
        file_path (str): dataset metadata file path.

    Return:
        (dict): dataset metadata information.
    """
    metadata = {}
    with open(file_path, 'r') as fd:
        for line in fd:
            key, value = line.strip().split(',')
            try:
                value = int(value.strip())
            except ValueError:
                pass
            metadata[key] = value
    return metadata


def write_dsmd(dsmd, file_path, auto_mkdirs=True):
    """ Write dataset metadata to given file path.

    Args:
        dsmd (dict): dataset metadata.
        file_path (str): file path to save dataset metadata.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.

    Return:
        (dict): dataset metadata information.
    """
    if auto_mkdirs:
        mv.mkdirs(os.path.dirname(file_path))

    with open(file_path, 'w') as fd:
        for key, value in dsmd.items():
            line = '%s, %s\n' % (str(key), str(value))
            fd.write(line)


def gen_cls_dsmd_file(root_dir, c2l_path, dsmd_path, classnames=None):
    """ Generate classification dataset metadata file from DataFolder for
    given classes.

    DataFolder is a directory structure for image classification problems.
    Each sub-directory contains images from a special class. DataFolder
    directory structure looks like
    -----------------------
    ├── class1
    │   ├── 1.png
    │   └── 2.png
    │   ...
    ├── class2
    │   ├── 3.png
    │   └── 4.png
    └── ...
    -----------------------

    Generated dsmd file looks like
    ----------
    |1.png, 0|
    |2.png, 0|
    |3.png, 1|
    |4.png, 1|
    |...     |
    ----------

    Args:
        root_dir (str): root data directory containing all the images.
        c2l_path (str): file path to save class2label info.
        dsmd_path (str): file path to save dataset metadata file.
        classnames (list[str]): names of given classes.
            If not given, all classes are considered.

    Note:
        Filename of each image in DataFolder should be unique. Otherwise,
        A FileExistsError will be thrown.
    """
    assert mv.isdir(root_dir)

    if classnames is None:
        classnames = os.listdir(root_dir)

    class2label = {}
    dsmd = {}

    for label, classname in enumerate(classnames):
        class2label[classname] = label

        class_dir = os.path.join(root_dir, classname)
        assert mv.isdir(class_dir)
        filenames = natsorted(os.listdir(class_dir))
        for filename in filenames:
            if filename in dsmd:
                raise FileExistsError('filename {} already exists'.format(filename))
            dsmd[filename] = label

    write_dsmd(class2label, c2l_path)
    write_dsmd(dsmd, dsmd_path)


def merge_datafolder(in_dir, out_dir, auto_mkdirs=True, classnames=None):
    """ Merge images in DataFolder to a single directory.

    DataFolder is described in 'gen_cls_dsmd_file()'. This function will
    make a copy of all files. Original DataFolder is left unchanged.

    Args:
        in_dir (str): given DataFolder root directory.
        out_dir (str): directory to save images in DataFolder.
        classnames (list[str]): names of given classes to be collected.
            If not given, all classes are considered.

    Note:
        Filename of each image in DataFolder should be unique. Otherwise,
        A FileExistsError will be thrown.

    """
    assert mv.isdir(in_dir)

    # clean output directory
    if auto_mkdirs:
        mv.mkdirs(os.path.dirname(out_dir))
    mv.empty_dir(out_dir)

    if classnames is None:
        classnames = os.listdir(in_dir)

    for classname in classnames:
        class_dir = mv.joinpath(in_dir, classname)
        assert mv.isdir(class_dir)
        filenames = natsorted(os.listdir(class_dir))
        mv.copyfiles(filenames, out_dir, src_root=class_dir, non_overwrite=True)


def split_dsmd_file(dsmd_filepath, datasplit, shuffle=True):
    """ Split a dataset medadata file into 3 parts.

    Split a dataset metadata file into 'train.txt', 'val.txt' and 'test.txt'.
    And put them in the same directory with given dsmd file.

    dsmd_filepath (str): file path of dataset metadata.
    datasplit (dict[str, float]): how to split the dataset. e.g.
        {'train': 0.9, 'val': 0.1, 'test': 0.0}
    shuffle (bool): whether to shuffle the dataset before splitting.

    Note:
        0.0 < datasplit['train'] + datasplit['val'] + datasplit['test'] <= 1.0
    """
    dsmd = load_dsmd(dsmd_filepath)
    num_total = len(dsmd)

    keys = list(dsmd.keys())
    if shuffle:
        random.shuffle(keys)

    sum_ratio = 0.0
    splits = {}
    for filetitle, ratio in datasplit.items():
        filepath = filetitle + '.txt'
        splits[filepath] = int(num_total * ratio)
        sum_ratio += ratio
    assert 0.0 < sum_ratio <= 1.0

    start_index = 0
    for filepath, num_cur_split in splits.items():
        end_index = start_index + num_cur_split

        start_index = np.clip(start_index, 0, num_total)
        end_index = np.clip(end_index, 0, num_total)

        keys_split = keys[start_index:end_index]
        keys_split = natsorted(keys_split)
        dsmd_split = {keys: dsmd[keys] for keys in keys_split}
        write_dsmd(filepath, dsmd_split)

        start_index = end_index
