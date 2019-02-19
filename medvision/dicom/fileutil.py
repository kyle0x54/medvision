import medvision as mv


def isdicom(path):
    """ Judge whether a given file is a valid dicom.

    Args:
        path(str): given file path.

    Returns:
        (bool): True if given file path is a valid dicom, otherwise False.
    """
    if not mv.isfile(path):
        return False

    # read preamble and magic code
    with open(path, 'rb') as f:
        header = f.read(132)

    if not header:
        return False

    # magic code of a dicom file should be 'DICM'
    magic_code = header[128:132]
    if magic_code != b'DICM':
        return False
    else:
        return True


def isdicomdir(path):
    """ Judge whether a given directory is a valid dicom directory.

    If given directory only contains dicoms (at least one dicom file),
    it is a dicom directory. Otherwise, it is not a dicom directory.

    Args:
        path(str): given directory path.

    Returns:
        (bool): True if given directory path is a dicom directory,
                otherwise False.
    """
    if not mv.isdir(path):
        return False

    for file_name in mv.listdir(path):
        file_path = mv.joinpath(path, file_name)
        if not isdicom(file_path):
            return False
    else:
        return True
