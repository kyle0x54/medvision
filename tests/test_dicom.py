import math

import numpy as np
import pytest

import medvision as mv

DATA_DIR = mv.joinpath(mv.parentdir(__file__), 'data')
DCM_PATH = mv.joinpath(DATA_DIR, 'dicoms', 'brain_001.dcm')


def gen_path(*paths):
    return mv.joinpath(DATA_DIR, *paths)


def assert_image_equal(a, b):
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    assert math.isclose(diff.max(), 0.0)


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
    return [float(s) for s in tag_str]


def test_dcmread():
    img = mv.dcmread(DCM_PATH)
    assert img.dtype == np.int16
    assert img.shape == (256, 256)

    img, info = mv.dcmread(DCM_PATH, read_header=True)
    assert img.dtype == np.int16
    assert img.shape == (256, 256)
    assert info.LargestImagePixelValue == 884
    assert info.PatientID == '123565'  # patient id
    assert info.StudyID == '8811'  # study id

    # pixel spacing
    assert tag2list(info.PixelSpacing) == [0.859375, 0.859375]


def test_dcminfo():
    info = mv.dcminfo(DCM_PATH)
    assert int(info.RescaleIntercept) == 0
    assert info.LargestImagePixelValue == 884
    assert info.PatientID == '123565'  # patient id
    assert info.StudyID == '8811'  # study id

    # pixel spacing
    assert tag2list(info.PixelSpacing) == [0.859375, 0.859375]


def test_dcmread_dr():
    img = mv.dcmread_dr(DCM_PATH)
    img_, ds = mv.dcmread(DCM_PATH, read_header=True)
    assert_image_equal(img, img_)

    img = mv.dcmread_dr(DCM_PATH, mv.DrReadMode.UNCHANGED)
    img_, ds = mv.dcmread(DCM_PATH, read_header=True)
    assert_image_equal(img, img_)

    img = mv.dcmread_dr(DCM_PATH, mv.DrReadMode.MONOCHROME1)
    img_, ds = mv.dcmread(DCM_PATH, read_header=True)
    assert_image_equal(img, img_.max() - img_)

    img, ds = mv.dcmread_dr(DCM_PATH,
                            mv.DrReadMode.MONOCHROME1,
                            read_header=True)
    assert ds.PhotometricInterpretation.find('MONOCHROME1') != -1

    img, ds = mv.dcmread_dr(DCM_PATH, read_header=True)
    assert ds.PhotometricInterpretation.find('MONOCHROME2') != -1

    img, ds = mv.dcmread_dr(DCM_PATH,
                            mv.DrReadMode.UNCHANGED,
                            read_header=True)
    assert ds.PhotometricInterpretation.find('MONOCHROME2') != -1


def test_dcmread_dr_itk():
    img = mv.dcmread_dr(DCM_PATH)
    img_itk = mv.dcmread_dr_itk(DCM_PATH)
    assert img_itk.dtype == img.dtype
    assert_image_equal(img, img_itk)

    img = mv.dcmread_dr(DCM_PATH, mv.DrReadMode.UNCHANGED)
    img_itk = mv.dcmread_dr_itk(DCM_PATH, mv.DrReadMode.UNCHANGED)
    assert_image_equal(img, img_itk)

    img = mv.dcmread_dr(DCM_PATH, mv.DrReadMode.MONOCHROME1)
    img_itk = mv.dcmread_dr_itk(DCM_PATH, mv.DrReadMode.MONOCHROME1)
    assert_image_equal(img, img_itk)

    img = mv.dcmread_dr(DCM_PATH, mv.DrReadMode.MONOCHROME2)
    img_itk = mv.dcmread_dr_itk(DCM_PATH, mv.DrReadMode.MONOCHROME2)
    assert_image_equal(img, img_itk)
