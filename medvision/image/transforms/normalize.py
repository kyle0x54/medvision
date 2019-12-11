import numpy as np
import medvision as mv


def normalize_grayscale(src, to_float=True):
    """ Rescale image intensity.

    Rescale an grayscale image's intensity range to [0.0, 1.0].

    Args:
        src (ndarray): image to be intensity rescaled.
        to_float (bool): whether to convert the input image to float
            before intensity rescale.

    Return:
        (ndarray): intensity rescaled image.
    """
    assert src.ndim == 2
    epsilon = 0.00001

    if to_float:
        src = src.astype(np.float32)

    min_val, max_val = np.min(src), np.max(src)
    if max_val - min_val < epsilon:
        max_val += epsilon

    return (src - min_val) / (max_val - min_val)


def normalize_rgb(img, mean, std):
    """ Normalize an image.

    Subtract mean per channel and divide by std per channel. (support
    grayscale image and RGB image).

    Args:
        img (ndarray): image to be normalized.
        mean (tuple[float] or float): mean values.
        std (tuple[float] or float): standard deviations.

    Return:
        (ndarray): the normalized RGB image.

    Note:
        For grayscale image, first rescale intensity to [0.0, 255.0].
        Then convert into a 3-channel RGB image before normalization.
    """
    img = img.astype(np.float32)

    if img.ndim == 2:
        img = normalize_grayscale(img, to_float=False) * 255.0
        img = mv.gray2rgb(img)

    return (img - mean) / std


def denormalize_rgb(img, mean, std):
    """ Inverse function of normalize_rgb().

    Restore a normalized image to its original state.
    """
    img = (img * std) + mean
    return img


def imadjust_grayscale(
    im: np.ndarray,
    low_pct: float = 0.01,
    high_pct: float = 0.99
):
    """ Increase contrast of a grayscale image.

    This function maps the intensity values in I to new values in J such that
    values between low_in and high_in map to values between 0 and 1.

    Args:
        im (np.ndarray): image to be enhanced.
        low_pct (float): mean values.
        high_pct (float): standard deviations.

    Return:
        (np.ndarray): the enhanced image.
    """
    assert im.ndim == 2
    assert 0.0 <= low_pct < high_pct <= 1.0

    low_loc = int(round((im.size - 1) * low_pct))
    high_loc = int(round((im.size - 1) * high_pct))

    im_flat = im.flatten()
    low_thr = im_flat[np.argpartition(im_flat, low_loc)[low_loc]]
    high_thr = im_flat[np.argpartition(im_flat, high_loc)[high_loc]]
    return mv.normalize_grayscale(np.clip(im, a_min=low_thr, a_max=high_thr))
