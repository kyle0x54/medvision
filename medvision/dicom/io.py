import pydicom


def dcmread(dicom_path, read_header=False):
    """ Read 2D image data from the DICOM file.

    Args:
        dicom_path (str): path of the dicom file.
        read_header (bool): whether to return the dicom header together
            with the image array.

    Return:
        (ndarray): dicom image array.
        (pydicom.dataset.FileDataset): an instance of FileDataset
        that represents a parsed DICOM file.
    """
    ds = pydicom.dcmread(dicom_path)
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
