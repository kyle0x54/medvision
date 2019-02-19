from enum import Enum, unique
import cv2
import medvision as mv


@unique
class ImreadMode(Enum):
    RGB = cv2.IMREAD_COLOR
    GRAY = cv2.IMREAD_GRAYSCALE
    UNCHANGED = cv2.IMREAD_UNCHANGED


def imread(file_path, flag=ImreadMode.RGB):
    """ Read an image.

    Args:
        file_path (str): image file path.
        flag (str): flags specifying the color type of a loaded image,
              refer to ImreadMode for more details.
    Returns:
        (ndarray): loaded image array.

    Note:
        The format of the loaded image array is RGB.
    """
    img = cv2.imread(file_path, flag.value)

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def imwrite(img, file_path, auto_mkdirs=True):
    """ Save image to specified file.

    Args:
        img (ndarray): image array to be written.
        file_path (str): specified file path to save to.
        auto_mkdirs (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.

    Returns:
        (bool): returns whether the image is saved successfully.

    Note:
        If the given image is a color image. It should be in RGB format.
    """
    if auto_mkdirs:
        mv.mkdirs(mv.parentdir(file_path))

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return cv2.imwrite(file_path, img)
