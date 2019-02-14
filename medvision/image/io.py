import cv2
from enum import Enum, unique, auto
from medvision.transforms import bgr2rgb
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
    """
    img = cv2.imread(file_path, flag.value)

    if flag == ImreadMode.COLOR:
        img = bgr2rgb(img)

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
    """
    if auto_mkdir:
        mkdirs(os.path.basename(file_path))
    return cv2.imwrite(file_path, img)
