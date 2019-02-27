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
    epsilon = np.finfo(np.float32).eps

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
