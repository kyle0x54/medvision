import datetime
import os
import re
import shutil
import uuid
from pathlib import Path
import platform
from typing import Union

import numpy as np
import pydicom
import SimpleITK as itk

import medvision as mv


def dcmread_pydicom(path: Union[str, Path], read_header: bool = False):
    """Read 2D image data from the DICOM file.

    Args:
        path (str or Path): path of the dicom file to be loaded.
        read_header (bool): whether to return the dicom header together
            with the image array.

    Return:
        (ndarray): dicom image array.
        (pydicom.dataset.FileDataset): an instance of FileDataset
            that represents a parsed DICOM file.
    """
    ds = pydicom.dcmread(str(path), force=True)
    img = ds.pixel_array

    if read_header:
        return img, ds
    else:
        return img


def dcminfo_pydicom(path: Union[str, Path]):
    """Read metadata (dicom tags) from DICOM file.

    Refer to pydicom.dcmread

    Args:
        path (str or Path): path of the dicom file.

    Return:
        (pydicom.dataset.FileDataset): an instance of FileDataset
        that represents a parsed DICOM file.
    """
    ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    return ds


def _get_itk_metadata(reader):
    """Read metadata from a SimpleITK.SimpleITK.ImageFileReader object.

    Args:
        reader (obj): SimpleITK file reader.

    Return:
        (dict): dicom tag and value pairs {tag: value}.
    """
    metadata = {}
    for tag in reader.GetMetaDataKeys():
        value = reader.GetMetaData(tag)
        metadata[tag] = value

    return metadata


def _get_itk_path(path: Union[str, Path]):
    if platform.system() != "Windows":
        return path

    # SimpleITK does not support path containing Chinese characters on Windows.
    # This is a tentative solution.
    if re.search("[\u4e00-\u9fff]", str(path)):
        medvision_dir = os.path.join(os.path.expanduser("~"), ".medvision")
        tmp_path = os.path.join(medvision_dir, str(uuid.uuid1()) + ".dcm")
        mv.mkdirs(medvision_dir)
        shutil.copyfile(path, tmp_path)
        return tmp_path
    else:
        return str(path)


def dcmread_itk(path: Union[str, Path], read_header: bool = False):
    """Read 2D image data and metadata from the DICOM file using SimpleITK.

    Args:
        dicom_path (str or Path): dicom file path.
        read_header (bool): whether to return the dicom header together
            with the image array.

    Return:
        (numpy.ndarray): dicom image array.
        (dict): The metadata stored in dicom tag and value pairs {tag: value}.
            For example, {'0008|0020': '20010316', '0018|0020': 'SE', ...}
    """
    itk_path = _get_itk_path(path)
    img_itk = itk.ReadImage(itk_path)
    img = itk.GetArrayFromImage(img_itk)
    img = np.squeeze(img)
    if read_header:
        metadata = _get_itk_metadata(img_itk)
        return img, metadata
    else:
        return img
