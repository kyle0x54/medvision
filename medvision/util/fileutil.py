from enum import Enum, unique, auto
from glob import glob as std_glob
import os
import shutil
from natsort import natsorted


isdir = os.path.isdir
isfile = os.path.isfile

joinpath = os.path.join
basename = os.path.basename

cp = shutil.copy
rm = os.remove
cptree = shutil.copytree
rmtree = shutil.rmtree
move = shutil.move


def mkdirs(path, mode=0o777):
    path = os.path.expanduser(path)
    os.makedirs(path, mode, exist_ok=True)


def empty_dir(path):
    assert isdir(path)
    rmtree(path)
    mkdirs(path)


def non_overwrite_cp(src, dst):
    if isfile(dst):
        raise FileExistsError('target file {} already exists'.format(dst))

    if isdir(dst):
        filename = basename(src)
        dst_filepath = joinpath(dst, filename)
        if isfile(dst_filepath):
            raise FileExistsError(
                'target file {} already exists'.format(dst_filepath))

    return cp(src, dst)


def copyfiles(src_paths, dst_dir, src_root=None, non_overwrite=False):
    assert isdir(dst_dir)
    assert isinstance(src_paths, (tuple, list))

    cp_func = non_overwrite_cp if non_overwrite else cp
    for src_path in src_paths:
        cp_func(joinpath(src_root, src_path), dst_dir)


@unique
class GlobMode(Enum):
    FILE = auto()
    DIR = auto()
    ALL = auto()


def glob(root, pattern, mode=GlobMode.FILE, recursive=False):
    root = os.path.expanduser(root)
    pattern = joinpath('**', pattern) if recursive else pattern
    pathname = joinpath(root, pattern)

    result = std_glob(pathname, recursive=recursive)

    if mode == GlobMode.FILE:
        result = filter(isfile, result)
    elif mode == GlobMode.DIR:
        result = filter(isdir, result)
    else:  # GlobMode.ALL
        pass

    return natsorted(result)
