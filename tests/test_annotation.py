import tempfile

import medvision as mv

DATA_DIR = mv.joinpath(mv.parentdir(__file__), 'data')
DCM_DIR = mv.joinpath(DATA_DIR, 'rws', 'dcms')
RWS_PATH = mv.joinpath(DATA_DIR, 'rws', 'brain_001.json')


def assert_rws_bbox_right(rws):
    assert rws['height'] == 256
    assert rws['width'] == 256
    assert mv.basename(rws['image_path']) == 'brain_001.dcm'
    assert len(rws['shapes']) == 1
    instance = rws['shapes'][0]
    assert instance['category'] == 'teeth'
    assert instance['bbox'] == [93, 3, 170, 60]


def test_load_rws_bbox():
    rws = mv.load_rws_bbox(RWS_PATH)
    assert_rws_bbox_right(rws)


def test_save_rws_bbox():
    bbox = [[93, 3, 170, 60]]
    height = 256
    width = 256
    filename = 'brain_001.dcm'
    category = 'teeth'

    with tempfile.TemporaryDirectory(prefix='medvision') as dirname:
        filepath = mv.joinpath(dirname, filename)
        mv.save_rws_bbox(filepath, bbox, (height, width), fixed_label=category)
        rws = mv.load_rws_bbox(filepath)
        assert_rws_bbox_right(rws)


def test_rws2dsmd_bbox():
    dsmd = mv.rws2dsmd_bbox([RWS_PATH], lambda x: 0, num_classes=1)
    assert len(dsmd) == 1
    assert 'brain_001' in dsmd
    assert dsmd['brain_001'][0][0].tolist() == [93, 3, 170, 60]

    mv.dsmd2rws_bbox(DCM_DIR, dsmd, {"teeth": 0}, suffix='.json_A1')
    rws_path = mv.joinpath(DCM_DIR, 'brain_001.json_A1')
    assert mv.isfile(rws_path)
    rws = mv.load_rws_bbox(rws_path)
    assert_rws_bbox_right(rws)
    mv.rm(rws_path)
