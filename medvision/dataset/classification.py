import ast

import medvision as mv


def load_cls_dsmd(dsmd_path):
    data = {}
    with open(dsmd_path, "r") as fd:
        for line in fd:
            key, value = line.strip().split(",", 1)
            try:  # try to interpret annotation as int or list[int].
                value = ast.literal_eval(value.strip())
            except (SyntaxError, ValueError):
                pass
            data[key] = value

    return mv.make_dsmd(data)


def save_cls_dsmd(dsmd_path, data, auto_mkdirs=True):
    if auto_mkdirs:
        mv.mkdirs(mv.parentdir(dsmd_path))

    dsmd = mv.make_dsmd(data)
    with open(dsmd_path, "w") as fd:
        for key, value in dsmd.items():
            if mv.isarrayinstance(value):  # handle multi-label case
                value = ",".join([str(entry) for entry in value])
            line = "%s,%s\n" % (str(key), str(value))
            fd.write(line)


def gen_cls_dsmd_file_from_datafolder(root_dir, c2l_path, dsmd_path, classnames=None):
    """Generate classification dataset metadata file from DataFolder for
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
        classnames = mv.listdir_natsorted(root_dir)

    class2label = {}
    dsmd = {}

    for label, classname in enumerate(classnames):
        class2label[classname] = label

        class_dir = mv.joinpath(root_dir, classname)
        assert mv.isdir(class_dir)
        filenames = mv.listdir_natsorted(class_dir)
        for filename in filenames:
            if filename in dsmd:
                raise FileExistsError("filename {} already exists".format(filename))
            dsmd[filename] = label

    mv.save_dsmd(c2l_path, class2label)
    mv.save_dsmd(dsmd_path, dsmd)


def gen_cls_ds_from_datafolder(in_dir, out_dir, auto_mkdirs=True, classnames=None):
    """Generate classification dataset from DataFolder.

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
        mv.mkdirs(out_dir)
    mv.empty_dir(out_dir)

    if classnames is None:
        classnames = mv.listdir_natsorted(in_dir)

    for classname in classnames:
        class_dir = mv.joinpath(in_dir, classname)
        assert mv.isdir(class_dir)
        filenames = mv.listdir_natsorted(class_dir)
        mv.copyfiles(filenames, out_dir, class_dir, non_overwrite=True)
