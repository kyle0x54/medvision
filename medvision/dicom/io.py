from pathlib import Path
from typing import Union
import pydicom
import SimpleITK as itk
import numpy as np


def dcmread(
    path: Union[str, Path],
    read_header: bool = False,
    itk_handler_enabled: bool = True
):
    """ Read 2D image data from the DICOM file.

    Args:
        path (str or Path): path of the dicom file to be loaded.
        read_header (bool): whether to return the dicom header together
            with the image array.
        itk_handler_enabled (bool): whether to use SimpleITK to read the dicom
            if pydicom fails.

    Return:
        (ndarray): dicom image array.
        (pydicom.dataset.FileDataset): an instance of FileDataset
            that represents a parsed DICOM file.

    N.B.
        If itk_handler is enabled, segmentation fault (caused by SimpleITK
        reader) might happen which cannot be caught in python code.
    """
    ds = pydicom.dcmread(str(path))
    if itk_handler_enabled:
        try:
            img = ds.pixel_array
        except Exception:
            img_itk = itk.ReadImage(str(path))
            img = itk.GetArrayFromImage(img_itk)
            img = np.squeeze(img)
    else:
        img = ds.pixel_array

    if read_header:
        return img, ds
    else:
        return img


def dcminfo(path: Union[str, Path]):
    """ Read metadata (dicom tags) from DICOM file.

    Refer to pydicom.dcmread

    Args:
        path (str or Path): path of the dicom file.

    Return:
        (pydicom.dataset.FileDataset): an instance of FileDataset
        that represents a parsed DICOM file.
    """
    ds = pydicom.dcmread(str(path))
    return ds
