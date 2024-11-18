import numpy as np


def normalize_image(src: np.ndarray):
    """Rescale grayscale image intensity.

    Rescale an grayscale image's intensity range to [0.0, 1.0].

    Args:
        src (ndarray): image to be intensity rescaled.

    Return:
        (ndarray): intensity rescaled image.
    """
    assert src.ndim == 2
    epsilon = 0.00001

    if src.dtype != np.float32:
        src = src.astype(np.float32)

    min_val, max_val = np.min(src), np.max(src)
    if max_val - min_val < epsilon:
        max_val += epsilon

    return (src - min_val) / (max_val - min_val)


def denormalize_image(
    img: np.ndarray,
    mean: float | list[float, float, float],
    std: float | list[float, float, float],
):
    """Inverse function of normalize().

    Restore a normalized image to its original state.
    """
    img = (img * std) + mean
    return img


def imadjust(
    im: np.ndarray,
    low_pct: float = 0.01,
    high_pct: float = 0.99,
    ma: np.ndarray | None = None,
):
    """Increase contrast of a grayscale image.

    This function maps the intensity values in I to new values in J such that
    values between low_in and high_in map to values between 0 and 1.

    Args:
        im (np.ndarray): The input grayscale image.
        low_pct (float): Lower percentile (e.g., 0.01 for 1%) for intensity mapping.
        high_pct (float): Upper percentile (e.g., 0.99 for 99%) for intensity mapping.
        ma (np.ndarray or None): Optional mask array. If specified, only non-zero pixels
            in the mask will be considered for intensity adjustment. 

    Return:
        (np.ndarray): the enhanced image.
    """
    assert im.ndim == 2
    assert 0.0 <= low_pct < high_pct <= 1.0

    if ma is not None:
        assert ma.shape == im.shape, "Mask shape must match image shape."

    low_loc = int(round((im.size - 1) * low_pct))
    high_loc = int(round((im.size - 1) * high_pct))
    im_flat = im.flatten() if ma is None else im[ma != 0]
    low_thr = im_flat[np.argpartition(im_flat, low_loc)[low_loc]]
    high_thr = im_flat[np.argpartition(im_flat, high_loc)[high_loc]]
    return normalize_image(np.clip(im, a_min=low_thr, a_max=high_thr))
