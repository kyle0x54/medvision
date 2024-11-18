import math

import cv2
import numpy as np
import pytest

import medvision as mv

DATA_DIR = mv.joinpath(mv.parentdir(__file__), 'data')
PNG_IMG_PATH = mv.joinpath(DATA_DIR, 'pngs', 'Blue-Ogi.png')
IM_GRAY = mv.imread(PNG_IMG_PATH, mv.ImreadMode.GRAY)
IM_RGB = mv.imread(PNG_IMG_PATH)


def assert_image_equal(a, b):
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    assert math.isclose(diff.max(), 0.0)


def test_normalize_grayscale():
    im_std = mv.normalize_image(IM_GRAY)
    assert math.isclose(im_std.max(), 1.0)
    assert math.isclose(im_std.min(), 0.0)


@pytest.mark.parametrize('img', [IM_GRAY, IM_RGB])
def test_imread_imwrite(img):
    dst_dir = mv.joinpath(DATA_DIR, 'temporary_subdir')
    dst_path = mv.joinpath(dst_dir, mv.basename(PNG_IMG_PATH))
    mv.mkdirs(dst_dir)

    ret_val = mv.imwrite(dst_path, img)
    assert ret_val
    img_reloaded = mv.imread(dst_path, mv.ImreadMode.UNCHANGED)
    assert_image_equal(img, img_reloaded)

    mv.rmtree(dst_dir)


def test_rgb2gray_gray2rgb():
    im_rgb = mv.gray2rgb(IM_GRAY)
    im_gray_2 = mv.rgb2gray(im_rgb)
    assert_image_equal(IM_GRAY, im_gray_2)

    im_gray_2 = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2GRAY)
    assert_image_equal(IM_GRAY, im_gray_2)


@pytest.mark.parametrize('img', [IM_GRAY, IM_RGB])
def test_vflip(img):
    assert_image_equal(mv.vflip(img), np.flipud(img))


@pytest.mark.parametrize('img', [IM_GRAY, IM_RGB])
def test_hflip(img):
    assert_image_equal(mv.hflip(img), np.fliplr(img))


@pytest.mark.parametrize('img', [IM_GRAY, IM_RGB])
def test_rot90(img):
    assert_image_equal(mv.rot90(img, 1), np.rot90(img, 1))
    assert_image_equal(mv.rot90(img, 2), np.rot90(img, 2))
    assert_image_equal(mv.rot90(img, 3), np.rot90(img, 3))
    assert_image_equal(mv.rot90(img, 4), img)

    assert_image_equal(mv.rot90(img, -1), np.rot90(img, -1))


@pytest.mark.parametrize('img', [IM_GRAY, IM_RGB])
def test_rotate(img):
    assert_image_equal(mv.rot90(img, 1), mv.rotate(img, 90))
    assert_image_equal(mv.rot90(img, 2), mv.rotate(img, 180))
    assert_image_equal(mv.rot90(img, 3), mv.rotate(img, 270))
    assert_image_equal(mv.rot90(img, 4), img)
    assert_image_equal(mv.rot90(img, -1), mv.rotate(img, -90))


@pytest.mark.parametrize('img', [IM_GRAY, IM_RGB])
def test_resize(img):
    im_resized_1 = mv.resize(img, 64, 32)
    im_resized_2 = cv2.resize(img, (32, 64))
    assert_image_equal(im_resized_1, im_resized_2)


@pytest.mark.parametrize('img', [IM_GRAY, IM_RGB])
def test_rescale(img):
    scale_factor = 0.5
    im_rescaled_1 = mv.rescale(img, scale_factor)
    im_rescaled_2 = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)
    assert_image_equal(im_rescaled_1, im_rescaled_2)

    scale_ = (32, 64)
    scale_factor = 32 / img.shape[0]
    im_rescaled_1 = mv.rescale(img, scale_)
    im_rescaled_2 = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)
    assert_image_equal(im_rescaled_1, im_rescaled_2)

    with pytest.raises(AssertionError):
        mv.rescale(img, 0)
        mv.rescale(img, (-2, 2))
        mv.rescale(img, (2, -2))


@pytest.mark.parametrize('img', [IM_GRAY, IM_RGB])
def test_crop(img):
    i, j, h, w = 1, 2, 27, 18
    roi = mv.crop(img, i, j, h, w)
    img_refill = img.copy()
    img_refill[i:i+h, j:j+w] = roi
    assert_image_equal(img, img_refill)


@pytest.mark.parametrize('img', [IM_GRAY, IM_RGB])
def test_center_crop(img):
    roi = mv.center_crop(img, 128, 128)
    assert_image_equal(img, roi)

    roi = mv.center_crop(img, 64, 64)
    roi_2 = mv.crop(img, 32, 32, 64, 64)
    assert_image_equal(roi, roi_2)


@pytest.mark.parametrize('img', [IM_GRAY, IM_RGB])
def test_pad_to_square(img):
    img_pad = mv.pad_to_square(img)
    assert_image_equal(img, img_pad)

    img_crop = mv.crop(img, 32, 0, 64, 128)
    img_pad = mv.pad_to_square(img_crop, border_mode=cv2.BORDER_CONSTANT)
    roi = mv.crop(img_pad, 32, 0, 64, 128)
    assert_image_equal(img_crop, roi)
    img_pad[32:32+64, :] = 0
    assert img_pad.max() == 0

    img_pad = mv.pad_to_square(img_crop)
    roi = mv.crop(img_pad, 32, 0, 64, 128)
    assert_image_equal(img_crop, roi)

    img_pad_topleft = mv.pad_to_square(
        img_crop, align_mode="topleft", border_mode=cv2.BORDER_CONSTANT, pad_value=0
    )
    assert img_pad_topleft[:img_crop.shape[0], :img_crop.shape[1]].max() == img_crop.max()  # 确保原图在左上角
    assert img_pad_topleft[:img_crop.shape[0], :img_crop.shape[1]].min() == img_crop.min()
    assert (img_pad_topleft[img_crop.shape[0]:, :] == 0).all()  # 确保底部填充值为 0
    assert (img_pad_topleft[:, img_crop.shape[1]:] == 0).all()  # 确保右侧填充值为 0
