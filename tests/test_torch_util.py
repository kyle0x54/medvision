import math
import numpy as np
import torch
import medvision as mv
import pytest


DATA_DIR = mv.joinpath(mv.parentdir(__file__), 'data')
PNG_IMG_PATH = mv.joinpath(DATA_DIR, 'pngs', 'Blue-Ogi.png')
IM_GRAY = mv.imread(PNG_IMG_PATH, mv.ImreadMode.GRAY)
IM_RGB = mv.imread(PNG_IMG_PATH)


def assert_image_equal(a, b):
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    assert math.isclose(diff.max(), 0.0)


@pytest.mark.parametrize('img', [IM_GRAY, IM_RGB])
def test_make_np(img):
    tensor = torch.from_numpy(img)
    img_converted = mv.make_np(tensor)
    assert_image_equal(img, img_converted)
