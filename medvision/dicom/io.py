import numpy as np
import SimpleITK as itk


def _get_metadata(reader):
    """ Read metadata from a SimpleITK.SimpleITK.ImageFileReader object.

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


def dcmread(dicom_path, read_header=False):
    """ Read 2D image data from the DICOM file.

    Args:
        dicom_path (str): path of the dicom file.
        read_header (bool): whether to return the dicom header together
            with the image array.

    Return:
        (ndarray): dicom image array.
        (dict, optional): dicom metadata.
    """
    img_itk = itk.ReadImage(dicom_path)
    img = itk.GetArrayFromImage(img_itk)
    img = np.squeeze(img)
    if read_header:
        metadata = _get_metadata(img_itk)
        return img, metadata
    else:
        return img


def dcminfo(dicom_path):
    """ Read metadata (dicom tags) from DICOM file.

    The metadata is stored in dicom tag and value pairs {tag: value}.
    For example, {'0008|0020': '20010316', '0018|0020': 'SE', ...}

    Args:
        dicom_path (str): path of the dicom file.

    Return:
        (dict): dicom metadata.
    """
    reader = itk.ImageFileReader()
    reader.SetFileName(dicom_path)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()

    metadata = _get_metadata(reader)
    return metadata
