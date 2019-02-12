from contextlib import contextmanager
import os
import medvision as mv
import pytest


DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


@contextmanager
def not_raises(exception):
    try:
        yield
    except exception:
        raise pytest.fail("DID RAISE {0}".format(exception))


def test_mkdirs():
    with not_raises(FileExistsError):
        mv.mkdirs(DATA_DIR)

    path = mv.joinpath(DATA_DIR, 'temporary_subdir')
    mv.mkdirs(path)
    assert mv.isdir(path)
    mv.rmtree(path)


def test_copyfiles():
    dst_dir = mv.joinpath(DATA_DIR, 'temporary_subdir')
    mv.mkdirs(dst_dir)

    src_paths = ['brain_001.dcm', 'brain_002.dcm']
    mv.copyfiles(src_paths, dst_dir, mv.joinpath(DATA_DIR, 'dicoms'))
    assert len(os.listdir(dst_dir)) == 2
    mv.rmtree(dst_dir)


def test_glob_file():
    root = DATA_DIR
    filepaths = mv.glob(root, '*.png', mode=mv.GlobMode.FILE, recursive=True)
    assert len(filepaths) == len(os.listdir(mv.joinpath(root, 'pngs')))

    root = DATA_DIR
    filepaths = mv.glob(root, '*.png', mode=mv.GlobMode.FILE, recursive=False)
    assert len(filepaths) == 0

    root = mv.joinpath(DATA_DIR, 'pngs')
    filepaths = mv.glob(root, '*.png', mode=mv.GlobMode.FILE, recursive=False)
    assert len(filepaths) == len(os.listdir(root))


def test_glob_dir():
    root = DATA_DIR
    filepaths = mv.glob(root, '*png*', mode=mv.GlobMode.DIR, recursive=True)
    assert len(filepaths) == 1

    root = DATA_DIR
    filepaths = mv.glob(root, '*', mode=mv.GlobMode.DIR, recursive=True)
    assert len(filepaths) == 3
