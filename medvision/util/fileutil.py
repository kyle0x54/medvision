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


def copyfiles(src_paths, dst_dir, src_root=None):
    assert isdir(dst_dir)
    assert isinstance(src_paths, (tuple, list))

    for src_path in src_paths:
        cp(joinpath(src_root, src_path), dst_dir)


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
