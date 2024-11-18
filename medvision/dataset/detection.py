from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd

import medvision as mv


def load_det_dsmd(dsmd_path: Union[str, Path], class2label: Union[str, Dict[str, int]]):
    """load detection dataset metadata.

    Args:
        dsmd_path (str or Path): dataset metadata file path.
        class2label (str or dict): class-to-label file.

    Return:
        (OrderedDict): Loaded dsmd is a OrderedDict looks like
        {
            data/1.png: [
                bboxes (ndarray) of category 'cat' of shape (n, 4) or (n, 5),
                bboxes (ndarray) of category 'dog' of shape (n, 4) or (n, 5),
                ...
            ]
            data/2.png: [
                ...
            ]
            ...
        }
    """
    if isinstance(class2label, str):
        class2label = mv.load_c2l(class2label)
    assert min(class2label.values()) == 0, "label should start from 0, but got %d" % min(
        class2label.values()
    )
    num_classes = len(class2label)

    df = pd.read_csv(dsmd_path, header=None)
    assert df.shape[1] == 6 or df.shape[1] == 7, "Incorrect dsmd file format %s" % dsmd_path

    data = {}
    for r in df.itertuples():
        filename = r[1]

        if filename not in data:
            empty_box = np.zeros((0, df.shape[1] - 2), dtype=np.float32)
            data[filename] = [empty_box] * num_classes

        if not pd.isnull(r[2]):
            box = r[2:-1]
            category_id = class2label[r[-1]]
            data[filename][category_id] = np.append(data[filename][category_id], [box], axis=0)

    return mv.make_dsmd(data)


def save_det_dsmd(
    dsmd_path: Union[str, Path],
    data: Dict[str, List[np.ndarray]],
    class2label: Union[str, Dict[str, int]],
    auto_mkdirs: bool = True,
):
    """Save dataset metadata to specified file.

    Args:
        dsmd_path (str or Path): file path to save dataset metadata.
        data (dict): dsmd to be serialized, refer to 'load_dsmd'.
        class2label (str or dict): class-to-label file or class2label dict.
        auto_mkdirs (bool): If the parent folder of `file_path` does not
            exist, whether to create it automatically.
    """
    if auto_mkdirs:
        mv.mkdirs(mv.parentdir(dsmd_path))

    # get label->class mapping
    if isinstance(class2label, str):
        class2label = mv.load_c2l(class2label)
    label2class = {value: key for key, value in class2label.items()}

    # write dataset metadata loop
    dsmd = mv.make_dsmd(data)
    data_list = []
    for filename, instance in dsmd.items():
        record = None
        for category_id, boxes in enumerate(instance):
            for box in boxes:
                record = [filename, *box.tolist(), label2class[category_id]]
                data_list.append(record)
        if record is None:
            data_list.append([filename])

    df = pd.DataFrame(data_list)
    df.to_csv(str(dsmd_path), header=False, index=False)


def merge_det_dsmds(ref_dsmd, *dsmds):
    """Merge dsmds into one dsmd.

    N.B. Overlapping boxes (even boxes with the same coordinates) are all kept.
    """
    filenames = set(ref_dsmd.keys())
    assert len(filenames) > 0
    for dsmd in dsmds:
        assert set(dsmd.keys()) == filenames, "dsmds not match"

    ref_filename = filenames.pop()
    num_categories = len(ref_dsmd[ref_filename])
    assert num_categories != 0
    filenames.add(ref_filename)

    res = {}
    for filename in filenames:
        res_instance = []
        for category_id in range(num_categories):
            instances = [dsmd[filename][category_id] for dsmd in dsmds] + [
                ref_dsmd[filename][category_id]
            ]
            res_instance.append(np.vstack(instances))
        res[filename] = res_instance

    return res
