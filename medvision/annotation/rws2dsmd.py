import numpy as np

import medvision as mv


def rws2dsmd_bbox(filepaths, class2label, suffix=".json", num_classes=None):
    """Convert rws bbox annotation to dsmd.

    Args:
        filepaths (str or list): file paths of rws annotation files or
            directory containing rws annotation files.
        class2label (str, dict or callable): class-to-label file,
            class2label dict or a callable. If label < 0, ignore.
        suffix (str): suffix of rws file. e.g. ".json", ".json_A1"
        num_classes (int or None): if class2label is a callable, num_classes
            should be explicitly specified.
    N.B.
        Key values of dsmd are file titles.
    """
    if callable(class2label):
        assert num_classes is not None, "must specify num_classes if class2label is a callable"
    else:
        if isinstance(class2label, str):
            class2label = mv.load_c2l(class2label)
        num_classes = len(class2label)

    if isinstance(filepaths, str):
        filepaths = mv.glob(filepaths, "*" + suffix)

    dsmd = {}
    for filepath in filepaths:
        key = mv.filetitle(filepath)
        dsmd[key] = [np.zeros((0, 4), dtype=np.float32)] * num_classes
        # to handle cases without annot when getting annot path from image path
        if not mv.isfile(filepath):
            continue
        shapes = mv.load_rws_bbox(filepath)["shapes"]
        for instance in shapes:
            try:
                label = class2label[instance["category"]]
            except TypeError:
                label = class2label(instance["category"])
            if label < 0:
                continue
            dsmd[key][label] = np.append(dsmd[key][label], [instance["bbox"]], axis=0)
    return mv.make_dsmd(dsmd)


def dsmd2rws_bbox(rws_dir, dsmd, class2label, suffix=".json_A1", score_thresh=0.0, dcm_dir=None):
    """Convert dsmd to rws bbox annotations.
    Args:
        rws_dir (str): directory to store converted rws file.
        dsmd (str or dsmd): file path of dsmd file or dsmd data.
        class2label (str or dict): class-to-label file or class2label dict.
        suffix (str): suffix of output rws file, For example, ".json_EA".
        score_thresh (float): threshold of confidence.
        dcm_dir (None or str): directory containing corresponding dicom files,
            if None, it is the same with rws_dir.
    N.B.
        1. Key values of dsmd are file titles.
        2. dicom file titles must be consistent with dsmd keys.
    """
    if isinstance(class2label, str):
        class2label = mv.load_c2l(class2label)
    label2class = {v: k for k, v in class2label.items()}

    if isinstance(dsmd, str):
        dsmd = mv.load_dsmd(dsmd, class2label, mode="det")

    if dcm_dir is None:
        dcm_dir = rws_dir
    dcm_titles = [mv.filetitle(path) for path in mv.glob(dcm_dir, "*.dcm")]
    assert set(dsmd.keys()) == set(dcm_titles)

    for key in dsmd.keys():
        shapes = []
        for i, bboxes in enumerate(dsmd[key]):
            for bbox in bboxes:
                if len(bbox) == 5 and bbox[-1] < score_thresh:
                    continue
                label = (
                    label2class[i]
                    if len(bbox) == 4
                    else label2class[i] + "_{:.3f}".format(bbox[-1])
                )
                shapes.append({"category": label, "bbox": bbox[:4].tolist()})
        if len(shapes) == 0:
            continue

        rws_path = mv.joinpath(rws_dir, key + suffix)
        ds = mv.dcminfo_pydicom(mv.joinpath(dcm_dir, key + ".dcm"))
        mv.save_rws_bbox(rws_path, shapes, (ds.Rows, ds.Columns))
