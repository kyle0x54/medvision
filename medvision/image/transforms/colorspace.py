import cv2


def bgr2gray(src, keepdim=False):
    """ Convert a BGR image to grayscale image.

    Args:
        src (ndarray): the input image.
        keepdim (bool): if False (by default), then return the
            single channel grayscale image, otherwise 3 channels.

    Returns:
        (ndarray): the converted grayscale image.
    """
    dst = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    return dst[..., None] if keepdim else dst


def rgb2gray(src, keepdim=False):
    """ Convert a RGB image to grayscale image.

    Args:
        src (ndarray): the input image.

    Returns:
        (ndarray): the converted grayscale image.
    """
    return cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)


def gray2bgr(src):
    """ Convert a grayscale image to BGR image.

    Args:
        src (ndarray or str): the input image.

    Returns:
        (ndarray): the converted BGR image.
    """
    src = src[..., None] if src.ndim == 2 else src
    return cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)


def gray2rgb(src):
    """ Convert a grayscale image to RGB image.

    Args:
        src (ndarray or str): the input image.

    Returns:
        (ndarray): the converted RGB image.
    """
    src = src[..., None] if src.ndim == 2 else src
    return cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)


def bgr2rgb(src):
    """ Convert a BGR image to RGB image.

    Args:
        src (ndarray or str): the input image.

    Returns:
        (ndarray): the converted RGB image.
    """
    return cv2.cvtColor(src, cv2.COLOR_BGR2RGB)


def rgb2bgr(src):
    """ Convert a RGB image to BGR image.

    Args:
        src (ndarray or str): the input image.

    Returns:
        (ndarray): the converted BGR image.
    """
    return cv2.cvtColor(src, cv2.COLOR_RGB2BGR)
