from enum import Enum, unique
import os
import cv2
from medvision.util import mkdirs


@unique
class ImreadMode(Enum):
    COLOR = cv2.IMREAD_COLOR
    GRAY = cv2.IMREAD_GRAYSCALE
    UNCHANGED = cv2.IMREAD_UNCHANGED


def imread(file_path, flag=ImreadMode.COLOR):
    """ Read an image.

    Args:
        file_path (str): image file path.
        flag (str): flags specifying the color type of a loaded image,
              refer to ImreadMode for more details.
    Returns:
        (ndarray): loaded image array.

    Note:
        If read color image, the returned image format is RGB.
    """
    img = cv2.imread(file_path, flag.value)

    if flag == ImreadMode.COLOR:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def imwrite(img, file_path, auto_mkdirs=True):
    """ Save image to specified file.

    Args:
        img (ndarray): image array to be written.
        file_path (str): specified file path to save to.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.

    Returns:
        (bool): Successful or not.

    Note:
        If the input image is a color image.The format should be RGB.
    """
    if auto_mkdirs:
        mkdirs(os.path.basename(file_path))

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return cv2.imwrite(file_path, img)
