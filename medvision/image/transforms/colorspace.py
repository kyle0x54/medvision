import cv2


def rgb2gray(src):
    """Convert a RGB image to grayscale image.

    Args:
        src (ndarray): the input image.

    Returns:
        (ndarray): the converted grayscale image.
    """
    return cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)


def gray2rgb(src):
    """Convert a grayscale image to RGB image.

    Args:
        src (ndarray): the input image.

    Returns:
        (ndarray): the converted RGB image.
    """
    return cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)
