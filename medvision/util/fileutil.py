import collections
import hashlib
import os
import shutil
from enum import Enum, auto, unique
from pathlib import Path

from natsort import natsorted

import medvision as mv

isdir = os.path.isdir
isfile = os.path.isfile


def listdir_natsorted(path):
    return natsorted(os.listdir(path))


joinpath = os.path.join
basename = os.path.basename
abspath = os.path.abspath
splitext = os.path.splitext


def parentdir(path):
    path = abspath(path)
    return os.path.dirname(path)


def filetitle(path):
    return os.path.splitext(os.path.basename(path))[0]


def change_suffix(path, new_suffix):
    return mv.splitext(path)[0] + new_suffix


cp = shutil.copy
rm = os.remove
cptree = shutil.copytree
rmtree = shutil.rmtree
move = shutil.move


def symlink(src, dst, overwrite=True, **kwargs):
    if os.path.lexists(dst) and overwrite:
        os.remove(dst)
    os.symlink(src, dst, **kwargs)


def mkdirs(path, mode=0o777):
    path = os.path.expanduser(path)
    path = abspath(path)
    os.makedirs(path, mode, exist_ok=True)


def empty_dir(path):
    assert isdir(path)
    rmtree(path)
    mkdirs(path)


def non_overwrite_cp(src, dst):
    if isfile(dst):
        raise FileExistsError("target file {} already exists".format(dst))

    if isdir(dst):
        filename = basename(src)
        dst_filepath = joinpath(dst, filename)
        if isfile(dst_filepath):
            raise FileExistsError("target file {} already exists".format(dst_filepath))

    return cp(src, dst)


def copyfiles(src_paths, dst_dir, src_root=None, non_overwrite=False):
    assert isdir(dst_dir)
    assert mv.isarrayinstance(src_paths)

    cp_func = non_overwrite_cp if non_overwrite else cp
    if src_root is not None:
        for src_path in src_paths:
            cp_func(joinpath(src_root, src_path), dst_dir)
    else:
        for src_path in src_paths:
            cp_func(src_path, dst_dir)


@unique
class GlobMode(Enum):
    FILE = auto()
    DIR = auto()
    ALL = auto()


def glob(root, pattern="*", mode=GlobMode.FILE, recursive=False):
    root = os.path.expanduser(root)
    root = os.path.abspath(root)
    root = Path(root)
    paths = root.rglob(pattern) if recursive else root.glob(pattern)
    paths = [str(entry) for entry in paths]

    if mode == GlobMode.FILE:
        paths = filter(isfile, paths)
    elif mode == GlobMode.DIR:
        paths = filter(isdir, paths)
    else:  # GlobMode.ALL
        pass

    return natsorted(paths)


def compute_md5_str(file_path):
    if not mv.isfile(file_path):
        return None

    with open(file_path, "rb") as f:
        m = hashlib.md5()
        m.update(f.read())
        md5_code = m.hexdigest()
        return str(md5_code).lower()


def find_duplicated_files(data_dir, pattern="*"):
    """Find duplicated files in specified directory.

    Args:
        data_dir (str): specified directory to be scanned.
        pattern: refer to 'glob()'.

    Return:
        (list[tuple]): duplicated file path pairs.
    """
    filepaths = glob(data_dir, pattern, mode=GlobMode.FILE, recursive=True)
    md5s = [compute_md5_str(filepath) for filepath in filepaths]
    md5_counts = collections.Counter(md5s)

    duplicated_files = []
    for key, count in md5_counts.items():
        if count > 1:
            candidates = tuple(filepaths[i] for i, x in enumerate(md5s) if x == key)
            duplicated_files.append(candidates)

    return duplicated_files
