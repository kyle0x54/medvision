import pytest

import medvision as mv

DATA_DIR = mv.joinpath(mv.parentdir(__file__), 'data')
DSMD_CLS_SL = mv.joinpath(DATA_DIR, 'texts', 'dsmd_cls_single_label.txt')
DSMD_CLS_ML = mv.joinpath(DATA_DIR, 'texts', 'dsmd_cls_multi_label.txt')
DF_DIR = mv.joinpath(DATA_DIR, 'datafolder')
DSMD_DF = mv.joinpath(DATA_DIR, 'texts', 'dsmd_datafolder.txt')
CLS2LBL = mv.joinpath(DATA_DIR, 'texts', 'class2label.txt')
DSMD_DET_GT = mv.joinpath(DATA_DIR, 'texts', 'dsmd_det_gt.csv')
DSMD_DET_DT = mv.joinpath(DATA_DIR, 'texts', 'dsmd_det_dt.csv')
DSMD_DET_C2L = mv.joinpath(DATA_DIR, 'texts', 'det_classes.csv')

RWS_DIR = mv.joinpath(DATA_DIR, 'rws')


def assert_equal_dsmds(a, b):
    assert len(a) == len(b)
    for key1, key2 in zip(a, b):
        assert key1 == key2
        assert a[key1] == a[key2]


@pytest.mark.parametrize('dsmd_file', [DSMD_CLS_SL, DSMD_CLS_ML])
def test_dsmd_io_cls(dsmd_file):
    tmp_dir = mv.joinpath(DATA_DIR, 'temporary_subdir')
    tmp_path = mv.joinpath(tmp_dir, 'tmp_dsmd.txt')

    dsmd_loaded = mv.load_dsmd(dsmd_file)
    mv.save_dsmd(tmp_path, dsmd_loaded)
    dsmd_reloaded = mv.load_dsmd(tmp_path)
    assert_equal_dsmds(dsmd_loaded, dsmd_reloaded)

    mv.rmtree(tmp_dir)


def test_dsmd_io_det():
    tmp_dir = mv.joinpath(DATA_DIR, 'temporary_subdir')
    tmp_path = mv.joinpath(tmp_dir, 'tmp_dsmd.txt')

    dsmd_loaded = mv.load_dsmd(DSMD_DET_GT, DSMD_DET_C2L, mode='det')
    mv.save_dsmd(tmp_path, dsmd_loaded, DSMD_DET_C2L, mode='det')
    dsmd_reloaded = mv.load_dsmd(DSMD_DET_GT, DSMD_DET_C2L, mode='det')
    assert_equal_dsmds(dsmd_loaded, dsmd_reloaded)

    mv.rmtree(tmp_dir)


def test_update_dsmd_keys():
    src_dsmd = mv.load_dsmd(DSMD_DET_GT, DSMD_DET_C2L, mode='det')
    dst_dsmd = mv.update_dsmd_keys(src_dsmd, parent_dir='/test', suffix='')
    assert mv.joinpath('/test', '000002.jpeg') in dst_dsmd
    dst_dsmd = mv.update_dsmd_keys(dst_dsmd, parent_dir=None)
    assert '000002' in dst_dsmd


def test_merge_det_dsmds():
    dsmd_dt = mv.load_det_dsmd(DSMD_DET_GT, DSMD_DET_C2L)
    assert len(dsmd_dt) == 703
    dst_dsmd = mv.merge_det_dsmds(dsmd_dt, dsmd_dt)
    assert len(dst_dsmd) == 703

    dsmd1 = mv.rws2dsmd_bbox(RWS_DIR, {'teeth': 0}, suffix='.json')
    dsmd2 = mv.rws2dsmd_bbox(RWS_DIR, {'teeth': 0}, suffix='.json_A1')
    dst_dsmd = mv.merge_det_dsmds(dsmd1, dsmd2)
    assert len(dst_dsmd) == 1
    assert len(dst_dsmd['brain_001']) == 1
    assert len(dst_dsmd['brain_001'][0]) == 2
    a1 = dst_dsmd['brain_001'][0][0]
    a2 = dst_dsmd['brain_001'][0][1]
    assert a1[0] in (90, 93)
    assert a2[1] in (3, 5)


def test_gen_cls_ds():
    tmp_dir = mv.joinpath(DATA_DIR, 'temporary_subdir')
    mv.mkdirs(tmp_dir)
    tmp_c2l_path = mv.joinpath(tmp_dir, 'tmp_c2l.txt')
    tmp_dsmd_path = mv.joinpath(tmp_dir, 'tmp_dsmd.txt')
    mv.gen_cls_dsmd_file_from_datafolder(DF_DIR, tmp_c2l_path, tmp_dsmd_path)

    dsmd = mv.load_dsmd(DSMD_DF)
    tmp_dsmd = mv.load_dsmd(tmp_dsmd_path)
    c2l = mv.load_dsmd(CLS2LBL)
    tmp_c2l = mv.load_dsmd(tmp_c2l_path)
    assert_equal_dsmds(dsmd, tmp_dsmd)
    assert_equal_dsmds(c2l, tmp_c2l)

    mv.empty_dir(tmp_dir)
    mv.gen_cls_ds_from_datafolder(DF_DIR, tmp_dir)
    assert len(mv.listdir_natsorted(tmp_dir)) == 8

    mv.rmtree(tmp_dir)


@pytest.mark.parametrize('dsmd_file', [DSMD_CLS_SL, DSMD_CLS_ML])
def test_split_dsmd_file(dsmd_file):
    tmp_dir = mv.joinpath(DATA_DIR, 'temporary_subdir')
    tmp_path = mv.joinpath(tmp_dir, 'tmp_dsmd.txt')
    mv.mkdirs(tmp_dir)
    mv.cp(dsmd_file, tmp_path)

    datasplit = {'train': 0.9, 'val': 0.1, 'test': 0.0}

    # shuffle
    mv.split_dsmd_file(tmp_path, datasplit)
    train_dsmd_file_path = mv.joinpath(tmp_dir, 'train.csv')
    val_dsmd_file_path = mv.joinpath(tmp_dir, 'val.csv')
    test_dsmd_file_path = mv.joinpath(tmp_dir, 'test.csv')
    assert mv.isfile(train_dsmd_file_path)
    assert mv.isfile(val_dsmd_file_path)
    assert not mv.isfile(test_dsmd_file_path)

    train_dsmd = mv.load_dsmd(train_dsmd_file_path)
    val_dsmd = mv.load_dsmd(val_dsmd_file_path)
    assert len(train_dsmd) == 18
    assert len(val_dsmd) == 2

    # non shuffle
    mv.split_dsmd_file(tmp_path, datasplit, shuffle=False)
    train_dsmd_file_path = mv.joinpath(tmp_dir, 'train.csv')
    val_dsmd_file_path = mv.joinpath(tmp_dir, 'val.csv')
    test_dsmd_file_path = mv.joinpath(tmp_dir, 'test.csv')
    assert mv.isfile(train_dsmd_file_path)
    assert mv.isfile(val_dsmd_file_path)
    assert not mv.isfile(test_dsmd_file_path)

    train_dsmd = mv.load_dsmd(train_dsmd_file_path)
    val_dsmd = mv.load_dsmd(val_dsmd_file_path)
    assert len(train_dsmd) == 18
    assert len(val_dsmd) == 2
    assert 'brain_001.dcm' in train_dsmd
    assert 'brain_019.dcm' in val_dsmd

    mv.rmtree(tmp_dir)
