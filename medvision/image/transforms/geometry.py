import cv2
import numpy as np

import medvision as mv


def vflip(img):
    """Flip an image vertically.

    Args:
        img (ndarray): image to be flipped.

    Returns:
        (ndarray): the vertically flipped image.
    """
    return np.ascontiguousarray(img[::-1, ...])


def hflip(img):
    """Flip an image horizontally.

    Args:
        img (ndarray): image to be flipped.

    Returns:
        (ndarray): the horizontally flipped image.
    """
    return np.ascontiguousarray(img[:, ::-1, ...])


def rot90(img, k):
    """Rotate 90 degrees.

    Rotate an array by 90 degrees for k times. Rotation direction is
    anticlockwise.

    Args:
        img (ndarray): image to be rotated.
        k (integer): number of times the array is rotated by 90 degrees.

    Returns:
        (ndarray): the rotated image.
    """
    return np.ascontiguousarray(np.rot90(img, k))


def rotate(src, angle, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101):
    """Rotate an image by arbitrarily degrees.

    Perform arbitrary rotations on an image. The rotation center is the
    geometric center of the image. The shape of the image keeps unchanged
    after the rotation.

    Args:
        src (ndarray): image to be rotated.
        angle (float): rotation angle in degrees, positive values mean
            anticlockwise rotation.
        interpolation (int): interpolation method (opencv).
        border_mode (int): border interpolation mode (opencv).

    Returns:
        (ndarray): the rotated image.
    """
    height, width = src.shape[:2]
    center = ((width - 1) * 0.5, (height - 1) * 0.5)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    dst = cv2.warpAffine(src, matrix, (width, height), flags=interpolation, borderMode=border_mode)
    return dst


def resize(img, height, width, interpolation=cv2.INTER_LINEAR):
    """Resize an image to the given size.

    Args:
        img (ndarray): the given image.
        height (int): target image height in pixel.
        width (int): target image width in pixel.
        interpolation (int): interpolation method (opencv).

    Returns:
        (ndarray): the resized image.
    """
    return cv2.resize(img, (width, height), interpolation=interpolation)


def rescale(src, scale, return_scale=False, interpolation=cv2.INTER_LINEAR):
    """Resize image while keeping the aspect ratio.

    Args:
        src (ndarray): the input image.
        scale (float or tuple[int]): the scaling factor or maximum size.
            If it is a float number, then the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, then the image will
            be rescaled as large as possible within the scale.
        return_scale (bool): whether to return the scaling factor besides the
            rescaled image.
        interpolation (int): interpolation method (opencv).

    Returns:
        (ndarray): the rescaled image
        (float, optional): the scaling factor
    """
    assert isinstance(scale, (float, int)) or mv.isarrayinstance(scale)

    height, width = src.shape[:2]

    # compute scale factor
    if isinstance(scale, (float, int)):
        assert scale > 0
    else:  # mv.isarrayinstance(scale):
        assert len(scale) == 2
        assert scale[0] > 0 and scale[1] > 0
        max_long_edge, max_short_edge = max(scale), min(scale)
        scale = min(max_long_edge / max(height, width), max_short_edge / min(height, width))

    # rescale the image
    dst_height, dst_width = round(height * scale), round(width * scale)
    dst = resize(src, dst_height, dst_width, interpolation)

    return (dst, scale) if return_scale else dst


def crop(img, i, j, h, w):
    """Crop an image.

    Args:
        img (numpy.ndarray): image to be cropped.
        i: upper pixel coordinate.
        j: left pixel coordinate.
        h: height of the cropped image.
        w: width of the cropped image.

    Returns:
        (ndarray): the cropped image.
    """
    return img[i : i + h, j : j + w, ...]


def center_crop(img, crop_height, crop_width):
    """Crop the central part of an image.

    Args:
        img (ndarray): image to be cropped.
        crop_height (int): height of the crop.
        crop_width (int): width of the crop.

    Return:
        (ndarray): the cropped image.
    """

    def get_center_crop_coords(height, width, crop_height, crop_width):
        y1 = (height - crop_height) // 2
        y2 = y1 + crop_height
        x1 = (width - crop_width) // 2
        x2 = x1 + crop_width
        return x1, y1, x2, y2

    height, width = img.shape[:2]
    x1, y1, x2, y2 = get_center_crop_coords(height, width, crop_height, crop_width)
    return img[y1:y2, x1:x2, ...]


def pad_to_square(
    src: np.ndarray,
    align_mode: str = "center",  # "center" or "topleft"
    border_mode: int = cv2.BORDER_REFLECT_101,
    pad_value: int = 0,
):
    """Pad an image to so that its height and width are the same.

    For example, an image with shape (3, 4) will be padded to (4, 4).

    Args:
        src (ndarray): image to be padded.
        border_mode (int): border interpolation mode (opencv).
        pad_value(int): values to be padded if using
            border_mode==cv2.BORDER_CONSTANT
    """
    height, width = src.shape[:2]

    if height == width:
        return src

    sz = max(height, width)

    top = abs(sz - height) // 2 if align_mode == "center" else 0
    bottom = abs(sz - height) - top
    left = abs(sz - width) // 2 if align_mode == "center" else 0
    right = abs(sz - width) - left

    if border_mode == cv2.BORDER_CONSTANT:
        dst = cv2.copyMakeBorder(src, top, bottom, left, right, border_mode, value=pad_value)
    else:
        dst = cv2.copyMakeBorder(src, top, bottom, left, right, border_mode)

    return dst
