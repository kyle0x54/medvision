from contextlib import contextmanager

import pytest

import medvision as mv

DATA_DIR = mv.joinpath(mv.parentdir(__file__), 'data')
DCM_DIR = mv.joinpath(DATA_DIR, 'dicoms')
PNG_DIR = mv.joinpath(DATA_DIR, 'pngs')


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
    mv.copyfiles(src_paths, dst_dir, DCM_DIR)
    assert len(mv.listdir_natsorted(dst_dir)) == 2

    with pytest.raises(FileExistsError):
        mv.non_overwrite_cp(mv.joinpath(DCM_DIR, src_paths[0]), dst_dir)

    with not_raises(FileExistsError):
        mv.copyfiles(src_paths, dst_dir, DCM_DIR, non_overwrite=False)

    with pytest.raises(FileExistsError):
        mv.copyfiles(src_paths, dst_dir, DCM_DIR, non_overwrite=True)

    mv.empty_dir(dst_dir)
    assert mv.isdir(dst_dir)
    assert len(mv.listdir_natsorted(dst_dir)) == 0
    mv.rmtree(dst_dir)


def test_glob_file():
    filepaths = mv.glob(DATA_DIR, '*.png', mode=mv.GlobMode.FILE,
                        recursive=True)
    assert len(filepaths) == 16

    filepaths = mv.glob(DATA_DIR, '*.png', mode=mv.GlobMode.FILE,
                        recursive=False)
    assert len(filepaths) == 0

    filepaths = mv.glob(PNG_DIR, mode=mv.GlobMode.FILE, recursive=False)
    assert len(filepaths) == len(mv.listdir_natsorted(PNG_DIR))


def test_glob_dir():
    root = DATA_DIR
    filepaths = mv.glob(root, '*png*', mode=mv.GlobMode.DIR, recursive=True)
    assert len(filepaths) == 1

    root = DATA_DIR
    filepaths = mv.glob(root, '*', mode=mv.GlobMode.DIR, recursive=True)
    assert len(filepaths) == 10


def test_has_duplicated_files():
    dst_dir = mv.joinpath(DATA_DIR, 'temporary_subdir')
    mv.mkdirs(dst_dir)

    # non duplicated files case
    src_paths = ['brain_001.dcm', 'brain_002.dcm', 'brain_003.dcm']
    mv.copyfiles(src_paths, dst_dir, DCM_DIR)
    assert len(mv.find_duplicated_files(dst_dir)) == 0

    # duplicated files case
    mv.non_overwrite_cp(mv.joinpath(DCM_DIR, src_paths[0]),
                        mv.joinpath(dst_dir, 'dup_0.dcm'))
    duplicated_files = mv.find_duplicated_files(dst_dir)
    assert len(duplicated_files) == 1
    assert (mv.joinpath(dst_dir, 'brain_001.dcm') in duplicated_files[0] and
            mv.joinpath(dst_dir, 'dup_0.dcm') in duplicated_files[0])

    mv.non_overwrite_cp(mv.joinpath(DCM_DIR, src_paths[1]),
                        mv.joinpath(dst_dir, 'dup_1.dcm'))
    duplicated_files = mv.find_duplicated_files(dst_dir)
    assert len(duplicated_files) == 2

    mv.rmtree(dst_dir)


def test_encrypt_decrypt():
    input = key = iv = b'testencryption'
    output = b'14\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00e' \
             b'\xd5?u\x18\x1d\xda;kB^\xef\x8c\xdc\xfa\x02'
    assert mv.encrypt(input, key, iv) == output
    assert mv.decrypt(output, key, iv) == input

    src_path = mv.joinpath(DCM_DIR, 'brain_001.dcm')
    with open(src_path, 'rb') as f:
        data_origin = f.read()
    data_encrypted = mv.encrypt(src_path, key, iv)
    data_decrypted = mv.decrypt(data_encrypted, key, iv)
    assert data_origin == data_decrypted
