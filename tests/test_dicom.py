import os
import numpy as np
import pytest
import medvision as mv


DATA_DIR = mv.joinpath(os.path.dirname(__file__), 'data')
DCM_PATH = mv.joinpath(DATA_DIR, 'dicoms', 'brain_001.dcm')


def gen_path(*paths):
    return mv.joinpath(DATA_DIR, *paths)


@pytest.mark.parametrize('given, expected', [
    (gen_path('dicoms', 'brain_001.dcm'), True),
    (gen_path('pngs', 'Blue-Ogi.png'), False),
    (gen_path('dicoms'), False),
    (gen_path('texts', 'null.txt'), False)
])
def test_isdicom(given, expected):
    assert mv.isdicom(given) == expected


@pytest.mark.parametrize('given, expected', [
    (gen_path('dicoms'), True),
    (gen_path('dicoms', 'brain_001.dcm'), False),
    (gen_path('pngs'), False),
    (gen_path('medvision'), False)
])
def test_isdicomdir(given, expected):
    assert mv.isdicomdir(given) == expected


def tag2list(tag_str):
    return [float(s) for s in tag_str.split('\\')]


def test_dcmread():
    img = mv.dcmread(DCM_PATH)
    assert img.dtype == np.int16
    assert img.shape == (256, 256)

    img, info = mv.dcmread(DCM_PATH, read_header=True)
    assert img.dtype == np.int16
    assert img.shape == (256, 256)
    assert int(info['0028|0107']) == 884
    assert info['0010|0020'] == '123565'  # patient id
    assert info['0020|0010'] == '8811'  # study id

    assert tag2list(info['0028|0030']) == [0.859375, 0.859375]  # pixel spacing


def test_dcminfo():
    info = mv.dcminfo(DCM_PATH)
    assert int(info['0028|1052']) == 0
    assert int(info['0028|0107']) == 884
    assert info['0010|0020'] == '123565'  # patient id
    assert info['0020|0010'] == '8811'  # study id
    assert tag2list(info['0028|0030']) == [0.859375, 0.859375]  # pixel spacing
