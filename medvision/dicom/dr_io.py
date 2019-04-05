from enum import Enum, unique
import medvision as mv


@unique
class DrReadMode(Enum):
    MONOCHROME1 = 1
    MONOCHROME2 = 2
    UNCHANGED = None


def _invert_if_needed(img, mode, mono):
    # Convert 'Photometric Interpretation' if needed
    mode_code = mode.value
    mono = 1 if mono.find('MONOCHROME1') != -1 else 2

    if mode_code != mono and mode_code is not DrReadMode.UNCHANGED.value:
        return img.max() - img
    else:
        return img


def dcmread_dr(dicom_path, mode=DrReadMode.MONOCHROME2, read_header=False):
    """ Read 2D digital radiography image data from the DICOM file.

    Args:
        dicom_path (str): path of the dicom file.
        mode ('DrReadMode'): read mode, refer to 'DrReadMode'.
        read_header (bool): whether to return the dicom header together
            with the image array.
    Returns:
        (ndarray): dicom image array.
        (dict, optional): dicom metadata.

    Note:
        MONOCHROME1 indicates that the greyscale ranges from bright to dark
        with ascending pixel values, whereas MONOCHROME2 ranges from dark
        to bright with ascending pixel values.
    """
    img, metadata = mv.dcmread(dicom_path, read_header=True)

    if img.ndim == 3:
        img = img[:, :, 0]

    # fetch monochrome value
    monochrome_tag = '0028|0004'
    assert monochrome_tag in metadata
    mono = metadata[monochrome_tag].upper()

    img = _invert_if_needed(img, mode, mono)

    if read_header:
        if mode != DrReadMode.UNCHANGED:
            metadata['0028|0004'] = mode.name
        return img, metadata
    else:
        return img
