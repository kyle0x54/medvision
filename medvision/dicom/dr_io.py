from enum import Enum, unique
from pathlib import Path
from typing import Union
import pydicom
import medvision as mv


@unique
class DrReadMode(Enum):
    MONOCHROME1 = 1
    MONOCHROME2 = 2
    UNCHANGED = None


def _invert_if_needed(img, mode, mono):
    # Convert 'Photometric Interpretation' if needed
    mono = 1 if mono.upper().find('MONOCHROME1') != -1 else 2
    if mode.value != mono and mode.value is not DrReadMode.UNCHANGED.value:
        return img.max() - img + img.min(), True
    else:
        return img, False


def dcmread_dr(
    path: Union[str, Path],
    mode: DrReadMode = DrReadMode.MONOCHROME2,
    read_header: bool = False,
    itk_handler_enabled: bool = True
):
    """ Read 2D digital radiography image data from the DICOM file.

    Args:
        path (str or Path): path of the dicom file.
        mode ('DrReadMode'): read mode, refer to 'DrReadMode'.
        read_header (bool): whether to return the dicom header together
            with the image array.
        itk_handler_enabled (bool): whether to use SimpleITK to read the
            dicom if pydicom fails. N.B. SimpleITK reader may cause
            segmentation fault which cannot be caught in python code.
    Returns:
        (ndarray): dicom image array.
        (dict, optional): dicom metadata.

    Note:
        MONOCHROME1 indicates that the greyscale ranges from bright to dark
        with ascending pixel values, whereas MONOCHROME2 ranges from dark
        to bright with ascending pixel values.
    """
    img, ds = mv.dcmread(
        path,
        read_header=True,
        itk_handler_enabled=itk_handler_enabled
    )

    if img.ndim == 3:
        img = img[:, :, 0]

    # fetch monochrome value
    assert 'PhotometricInterpretation' in ds
    mono = ds.PhotometricInterpretation

    img, is_inverted = _invert_if_needed(img, mode, mono)

    if read_header:
        if 'WindowCenter' in ds and 'WindowWidth' in ds:
            if isinstance(ds.WindowCenter, pydicom.multival.MultiValue):
                ds.WindowCenter = ds.WindowCenter[0]
                ds.WindowWidth = ds.WindowWidth[0]
            ds.WindowCenter = float(ds.WindowCenter)
            ds.WindowWidth = float(ds.WindowWidth)
            if is_inverted:
                ds.WindowCenter = (
                    float(img.max()) + float(img.min()) - ds.WindowCenter
                )
        if is_inverted:
            ds.PhotometricInterpretation = mode.name
        return img, ds
    else:
        return img
