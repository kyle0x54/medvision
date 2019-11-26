import pydicom
import SimpleITK as itk
import numpy as np


def dcmread(dicom_path, read_header=False, itk_handler_enabled=True):
    """ Read 2D image data from the DICOM file.

    Args:
        dicom_path (str): path of the dicom file.
        read_header (bool): whether to return the dicom header together
            with the image array.
        itk_handler_enabled (bool): whether to use SimpleITK to read the dicom
            if pydicom fails. N.B. SimpleITK reader may cause segmentation
            fault which cannot be caught in python code.

    Return:
        (ndarray): dicom image array.
        (pydicom.dataset.FileDataset): an instance of FileDataset
        that represents a parsed DICOM file.
    """
    ds = pydicom.dcmread(dicom_path)
    if itk_handler_enabled:
        try:
            img = ds.pixel_array
        except Exception:
            img_itk = itk.ReadImage(dicom_path)
            img = itk.GetArrayFromImage(img_itk)
            img = np.squeeze(img)
    else:
        img = ds.pixel_array

    if read_header:
        return img, ds
    else:
        return img


def dcminfo(dicom_path):
    """ Read metadata (dicom tags) from DICOM file.

    Refer to pydicom.dcmread

    Args:
        dicom_path (str): path of the dicom file.

    Return:
        (pydicom.dataset.FileDataset): an instance of FileDataset
        that represents a parsed DICOM file.
    """
    ds = pydicom.dcmread(dicom_path)
    return ds
