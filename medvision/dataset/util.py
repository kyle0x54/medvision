import ast
import collections
import random
from natsort import natsorted
import numpy as np
import medvision as mv


def load_dsmd(file_path):
    """ Load dataset metadata.

    Dataset metadata is a key-value pairs describing a dataset. For example,
    a dataset metadata looks like {'data/1.png': 1, 'data/2.png': 0, ...}.
    A dataset metadata file is a structured text file. For example,
    A single label classification dataset metadata file looks like.
    --------------
    |data/1.png,1|
    |data/2.png,0|
    |...         |
    --------------
    A multi label classification dataset metadata file looks like.
    ------------------
    |data/1.png,1,0,1|
    |data/2.png,0,0,0|
    |...             |
    ------------------
    A segmentation dataset metadata file looks like.
    ------------------------
    |data/1.png,data/1.mask|
    |data/2.png,data/2.mask|
    |...                   |
    ------------------------

    Args:
        file_path (str): dataset metadata file path.

    Return:
        (dict): dataset metadata information.
    """
    metadata = collections.OrderedDict()
    with open(file_path, 'r') as fd:
        for line in fd:
            key, value = line.strip().split(',', 1)
            # try to interpret annotation as reasonable type.
            try:
                value = ast.literal_eval(value.strip())
            except (SyntaxError, ValueError):
                pass
            metadata[key] = value
    return metadata


def load_dsmd_det(file_path, c2l_path):
    """
    A detection dataset metadata file looks like.
    --------------------------------
    |data/1.png,170,146,397,681,cat|
    |data/1.png,473,209,723,673,cat|
    |data/1.png,552,167,745,272,dog|
    |data/2.png,578,267,624,304,dog|
    |data/3.png,,,,,               |
    |...                           |
    --------------------------------
    or
    ------------------------------------
    |data/1.png,170,146,397,681,0.5,cat|
    |data/1.png,473,209,723,673,0.2,cat|
    |data/1.png,552,167,745,272,0.7,dog|
    |data/2.png,578,267,624,304,0.9,dog|
    |...                               |
    ------------------------------------
    Each line contains following information.
    (filepath,xmin,ymin,xmax,ymax,probability_score[optional], label)

    A class-to-label file looks like (note that the label starts from 0)
    --------
    |cat, 0|
    |dog, 1|
    |...   |
    --------

    Loaded dsmd is a dictionary looks like
    {
        data/1.png: [
            [bboxes of category 'cat' in ndarray of shape (n, 4)],
            [bboxes of category 'dog' in ndarray of shape (n, 4)],
            ...
        ]
        data/2.png: [
            ...
        ]
        ...
    }
    """
    # TODO: merge with 'load_dsmd'
    metadata = {}
    class2label = load_dsmd(c2l_path)
    num_classes = len(class2label)

    # label should start from 0
    assert min(class2label.values()) == 0

    with open(file_path, 'r') as fd:
        for line in fd:
            key, value = line.strip().split(',', 1)
            value, class_name = value.strip().rsplit(',', 1)
            label = class2label[class_name] if class_name else None

            if label is None:
                metadata[key] = [None] * num_classes
                continue

            # try to interpret annotation as reasonable type.
            try:
                value = ast.literal_eval(value.strip())
            except (SyntaxError, ValueError):
                pass

            if key not in metadata:
                metadata[key] = [None] * num_classes
                metadata[key][label] = [value]
            elif metadata[key][label] is None:
                metadata[key][label] = [value]
            else:
                metadata[key][label].append(value)

    # convert bboxes from list of lists to ndarray
    for key in metadata:
        for j in range(num_classes):
            if metadata[key][j] is not None:
                metadata[key][j] = np.array(metadata[key][j],
                                            dtype=np.float32)
            else:
                metadata[key][j] = np.zeros((0, 4), dtype=np.float32)

    metadata = collections.OrderedDict(natsorted(metadata.items()))
    return metadata


def save_dsmd(dsmd, file_path, auto_mkdirs=True):
    """ Save dataset metadata to specified file.

    Args:
        dsmd (dict): dataset metadata.
        file_path (str): file path to save dataset metadata.
        auto_mkdirs (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.
    """
    if auto_mkdirs:
        mv.mkdirs(mv.parentdir(file_path))

    ordered_dsmd = collections.OrderedDict(natsorted(dsmd.items()))
    with open(file_path, 'w') as fd:
        for key, value in ordered_dsmd.items():
            if mv.isarrayinstance(value):  # for multi label case
                value = ','.join([str(entry) for entry in value])
            line = '%s,%s\n' % (str(key), str(value))
            fd.write(line)


def save_dsmd_det(dsmd, file_path, label2class, auto_mkdirs=True):
    # TODO: merge with 'save_dsmd'
    if auto_mkdirs:
        mv.mkdirs(mv.parentdir(file_path))

    ordered_dsmd = collections.OrderedDict(natsorted(dsmd.items()))
    with open(file_path, 'w') as fd:
        for key, value in ordered_dsmd.items():
            if sum([len(i) for i in value]) == 0:
                fd.write('%s,,,,,\n' % str(key))
                continue

            for label, bboxes in enumerate(value):
                class_name = label2class[label]
                for bbox in bboxes:
                    bbox = ','.join([str(elem) for elem in bbox])
                    line = '%s,%s,%s\n' % (str(key), str(bbox), class_name)
                    fd.write(line)


def gen_cls_dsmd_file_from_datafolder(
        root_dir, c2l_path, dsmd_path, classnames=None):
    """ Generate classification dataset metadata file from DataFolder for
    specified classes.

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
    |1.png,0|
    |2.png,0|
    |3.png,1|
    |4.png,1|
    |...     |
    ----------

    Args:
        root_dir (str): root data directory containing all the images.
        c2l_path (str): file path to save class2label info.
        dsmd_path (str): file path to save dataset metadata file.
        classnames (list[str]): names of specified classes.
            If not given, all classes are considered.

    Note:
        This function is expected to be used together with
        'gen_cls_ds_from_datafolder()'.
        Filename of each image in DataFolder should be unique. Otherwise,
        A FileExistsError will be thrown.
    """
    assert mv.isdir(root_dir)

    if classnames is None:
        classnames = mv.listdir(root_dir)

    class2label = {}
    dsmd = {}

    for label, classname in enumerate(classnames):
        class2label[classname] = label

        class_dir = mv.joinpath(root_dir, classname)
        assert mv.isdir(class_dir)
        filenames = mv.listdir(class_dir)
        for filename in filenames:
            if filename in dsmd:
                raise FileExistsError(
                    'filename {} already exists'.format(filename))
            dsmd[filename] = label

    save_dsmd(class2label, c2l_path)
    save_dsmd(dsmd, dsmd_path)


def gen_cls_ds_from_datafolder(
        in_dir, out_dir, auto_mkdirs=True, classnames=None):
    """ Generate classification dataset from DataFolder.

    This function will make a copy of each image in the DataFolder to the
    specified directory. Original DataFolder is left unchanged.

    Args:
        in_dir (str): DataFolder root directory.
        out_dir (str): directory to save all the images in DataFolder.
        auto_mkdirs (bool): If `out_dir` does not exist, whether to create
            it automatically.
        classnames (list[str]): names of specified classes to be collected.
            If not given, all classes are considered.

    Note:
        This function is expected to be used together with
        gen_cls_dsmd_file_from_datafolder().
        Filename of each image in DataFolder should be unique. Otherwise,
        A FileExistsError will be thrown.
        DataFolder is described in 'gen_cls_dsmd_file_from_datafolder()'.
    """
    assert mv.isdir(in_dir)

    # clean output directory
    if auto_mkdirs:
        mv.mkdirs(mv.parentdir(out_dir))
    mv.empty_dir(out_dir)

    if classnames is None:
        classnames = mv.listdir(in_dir)

    for classname in classnames:
        class_dir = mv.joinpath(in_dir, classname)
        assert mv.isdir(class_dir)
        filenames = mv.listdir(class_dir)
        mv.copyfiles(filenames, out_dir, class_dir, non_overwrite=True)


def split_dsmd_file(dsmd_filepath, datasplit, shuffle=True):
    """ Split a dataset medadata file into 3 parts.

    Split a dataset metadata file into 'train.txt', 'val.txt' and 'test.txt'.
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

    dsmd = load_dsmd(dsmd_filepath)
    num_total = len(dsmd)

    keys = list(dsmd.keys())
    if shuffle:
        random.shuffle(keys)

    sum_ratio = 0.0
    splits = {}
    for filetitle, ratio in datasplit.items():
        filepath = mv.joinpath(dsmd_dir, filetitle + '.txt')
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
        if len(dsmd_split) != 0:
            save_dsmd(dsmd_split, filepath)

        start_index = end_index


# TODO: move to unit test
if __name__ == '__main__':
    file_path = '/home/huiying/Desktop/train_with_invert.csv'
    out_path = '/home/huiying/Desktop/out.csv'
    class2label_file_path = '/home/huiying/Desktop/classes.csv'
    dsmd = load_dsmd_det(file_path, class2label_file_path)

    class2label = load_dsmd(class2label_file_path)
    label2class = {value: key for key, value in class2label.items()}
    save_dsmd_det(dsmd, out_path, label2class, auto_mkdirs=True)

    dsmd = load_dsmd_det(out_path, class2label_file_path)
    dsmd = [value for key, value in dsmd.items()]
    dsmd_det = []
    for img_id, _ in enumerate(dsmd):
        dsmd_det.append([])
        for label_id, _ in enumerate(dsmd[img_id]):
            dsmd_det[img_id].append([])
            num_bboxes = len(dsmd[img_id][label_id])
            scores = np.ones((num_bboxes, 1), dtype=np.float32)
            dsmd_det[img_id][label_id] = np.hstack([dsmd[img_id][label_id],
                                                    scores])

    det_metric = mv.eval_det(dsmd, dsmd_det)
    print(det_metric)
