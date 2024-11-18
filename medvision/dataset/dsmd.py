from .classification import load_cls_dsmd, save_cls_dsmd
from .detection import load_det_dsmd, save_det_dsmd


def load_dsmd(dsmd_path, class2label=None, mode="cls"):
    """Load dataset metadata.

    Dataset Metadata is a key-value pairs describing a dataset.
    For example, a dataset metadata can be a dictionary looks like
    {
        'data/1.png': 1,
        'data/2.png': 0,
        ...
    }

    A dataset metadata file defines the mapping from files
    to annotations. Several types of data metadata files are listed below.

    +-------------------------------------------------------+
    | A. Single Label Classification Dataset Metadata File  |
    +-------------------------------------------------------+
    |data/1.png,1                                           |
    |data/2.png,0                                           |
    |...                                                    |
    +-------------------------------------------------------+

    +-------------------------------------------------------+
    | B. Multi-Label Classification Dataset Metadata File   |
    +-------------------------------------------------------+
    |data/1.png,1,0,1                                       |
    |data/2.png,0,1,0                                       |
    |...                                                    |
    +-------------------------------------------------------+

    +-------------------------------------------------------+
    | C. Segmentation Dataset Metadata File                 |
    +-------------------------------------------------------+
    |data/1.png,data/1_mask.png                             |
    |data/2.png,data/2_mask.png                             |
    |...                                                    |
    +-------------------------------------------------------+

    +-------------------------------------------------------+
    | D. Detection Dataset Metadata File A (bbox only)      |
    +-------------------------------------------------------+
    |data/1.png,170,146,397,681,cat                         |
    |data/1.png,473,209,723,673,cat                         |
    |data/1.png,552,167,745,272,dog                         |
    |data/2.png,578,267,624,304,dog                         |
    |data/3.png,,,,,                                        |
    |...                                                    |
    +-------------------------------------------------------+

    +-------------------------------------------------------+
    | E. Detection Dataset Metadata File B (bbox with score)|
    +-------------------------------------------------------+
    |data/1.png,170,146,397,681,0.5,cat                     |
    |data/1.png,473,209,723,673,0.2,cat                     |
    |data/1.png,552,167,745,272,0.7,dog                     |
    |data/2.png,578,267,624,304,0.9,dog                     |
    |data/3.png,,,,,                                        |
    |...                                                    |
    +-------------------------------------------------------+

    Args:
        dsmd_path (str): dataset metadata file path.
        class2label (str or dict): class-to-label file or dict.
        mode (str): dataset mission, can be one of 'cls', 'seg', 'det'.

    Return:
        (OrderedDict): dataset metadata.
    """
    if mode in ["cls", "seg"]:
        return load_cls_dsmd(dsmd_path)
    elif mode == "det":
        return load_det_dsmd(dsmd_path, class2label)
    else:
        raise ValueError("only support cls, seg, det modes")


def save_dsmd(dsmd_path, data, class2label=None, auto_mkdirs=True, mode="cls"):
    """Save dataset metadata to specified file.

    Args:
        dsmd_path (str): file path to save dataset metadata.
        data (dict): dataset metadata, refer to 'load_dsmd'.
        class2label (str or dict): class-to-label file or class2label dict.
        auto_mkdirs (bool): If the parent folder of `file_path` does
            not exist, whether to create it automatically.
        mode (str): dataset mission, can be one of 'cls', 'seg', 'det'.
    """
    if mode in ["cls", "seg"]:
        return save_cls_dsmd(dsmd_path, data, auto_mkdirs)
    elif mode == "det":
        return save_det_dsmd(dsmd_path, data, class2label, auto_mkdirs)
    else:
        raise ValueError("only support cls, seg, det modes")


def load_c2l(c2l_path):
    """Load class-to-label mapping.

    A class-to-label file defines the mapping from class_names to
    labels, which looks like (Note that the label value starts from 0)

    +------------------------------------------------------+
    | Class-to-Label File                                  |
    +------------------------------------------------------+
    |cat, 0                                                |
    |dog, 1                                                |
    |...                                                   |
    +------------------------------------------------------+

    Args:
        c2l_path (str): class-to-label file.

    Return:
        (OrderedDict): class-to-label mapping.
    """
    return load_dsmd(c2l_path)
