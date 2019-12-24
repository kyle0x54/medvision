import numpy as np
import medvision as mv


def rws2dsmd_bbox(
    filepaths,
    class2label=None,
    suffix='.json',
    num_classes=None
):
    """ Convert rws bbox annotation to dsmd.

    Args:
        filepaths (str or list): file paths of rws annotation files or
            directory containing rws annotation files.
        class2label (str, dict or callable): class-to-label file,
            class2label dict or a callable.
        suffix (str): suffix of rws file. e.g. ".json", ".json_A1"
        num_classes (int or None): if class2label is a callable, num_classes
            should be explicitly specified.
    N.B.
        dsmd key value is file title.
    """
    if callable(class2label):
        assert num_classes is not None,\
            'must specify num_classes if class2label is a callable'
    else:
        if isinstance(class2label, str):
            class2label = mv.load_c2l(class2label)
        num_classes = len(class2label)

    if isinstance(filepaths, str):
        filepaths = mv.glob(filepaths, '*' + suffix)

    dsmd = {}
    for filepath in filepaths:
        key = mv.filetitle(filepath)
        dsmd[key] = [np.zeros((0, 4), dtype=np.float32)] * num_classes
        # to handle cases without annot when getting annot path from image path
        if not mv.isfile(filepath):
            continue
        shapes = mv.load_rws_bbox(filepath)['shapes']
        for classname, box in shapes:
            try:
                label = class2label[classname]
            except TypeError:
                label = class2label(classname)
            dsmd[key][label] = np.append(dsmd[key][label], [box], axis=0)
    return mv.make_dsmd(dsmd)


def dsmd2rws_bbox(
    dsmd,
    dcm_dir,
    class2label=None,
    suffix='.json_A1',
    score_thresh=0.3
):
    """ Convert dsmd to rws bbox annotations.
    Args:
        dsmd (str or dsmd): file path of dsmd file or dsmd data.
        dcm_dir (str): dicom directory.
        class2label (str or dict): class-to-label file or class2label dict.
        suffix (str): suffix of output rws file, For example, ".json_EA".
        score_thresh (float): threshold of confidence.
    N.B.
        dsmd key value is file title.
    """
    if isinstance(class2label, str):
        class2label = mv.load_c2l(class2label)
    label2class = {v: k for k, v in class2label.items()}

    if isinstance(dsmd, str):
        dsmd = mv.load_dsmd(dsmd, class2label, mode='det')

    dcm_paths = mv.glob(dcm_dir, "*.dcm")
    assert len(dcm_paths) == len(dsmd)

    for key in dsmd.keys():
        shapes = []
        for i, bboxes in enumerate(dsmd[key]):
            for bbox in bboxes:
                if len(bbox) == 5 and bbox[-1] < score_thresh:
                    continue
                label = label2class[i] if len(bbox) == 4 else \
                    label2class[i] + "_{:.3f}".format(bbox[-1])
                shapes.append((label, bbox[:4].tolist()))
        if len(shapes) == 0:
            continue

        rws_path = mv.joinpath(dcm_dir, key + suffix)
        ds = mv.dcminfo(mv.joinpath(dcm_dir, key + '.dcm'))
        mv.save_rws_bbox(rws_path, shapes, (ds.Rows, ds.Columns))


if __name__ == '__main__':
    dsmd = rws2dsmd_bbox(
        ['0.json'],
        lambda x: 0,
        num_classes=1
    )
    dsmd2rws_bbox(dsmd, "dcm_dir", {"classname": 0})
    print(dsmd)
