import json
import numpy as np
import medvision as mv


def rws2dsmd_bbox(filepaths, num_classes, class2label):
    """ Convert rws bbox annotation to dsmd.

    Args:
        filepaths (str or list): file paths of rws annotation files or
            directory containing rws annotation files.
        num_classes (int): number of classes.
        class2label (str or callable): class-to-label file or class2label dict.

    N.B.
        dsmd key value is file title.
    """
    if isinstance(class2label, str):
        class2label = mv.load_c2l(class2label)

    if isinstance(filepaths, str):
        filepaths = mv.glob(filepaths, '*.json')

    dsmd = {}
    for filepath in filepaths:
        key = mv.filetitle(filepath)
        dsmd[key] = [None] * num_classes
        # to handle cases without annot when getting annot path from image path
        if not mv.isfile(filepath):
            continue
        shapes = mv.load_rws_bbox(filepath)['shapes']
        for label, bbox in shapes:
            label = class2label(label)
            bbox = np.array(bbox)
            if dsmd[key][label] is None:
                dsmd[key][label] = [bbox]
            else:
                dsmd[key][label].append(bbox)
    dsmd = mv.convert_bboxes_format(dsmd)
    dsmd = mv.make_dsmd(dsmd)
    return dsmd


def _gen_rws_shape(bbox, label):
    flags = {}
    shape_type = 'rectangle'
    return dict(
        label=label,
        points=[[float(bbox[0]), float(bbox[1])],
                [float(bbox[2]), float(bbox[3])]],
        shape_type=shape_type,
        flags=flags
    )


def dsmd2rws_det(dsmd, dcm_dir, suffix='.json_AI', class2label=None, thr=0.3):
    """ Convert dsmd to rws bbox annotations.
    Args:
        dsmd (str or dsmd): file path of dsmd file or dsmd data.
        dcm_dir (str): dicom directory.
        suffix (str): suffix of output rws file, For example, ".json_EA".
        class2label (str or dict): class-to-label file or class2label dict.
        thr (float): threshold of confidence.
    N.B.
        dsmd key value is file title.
    """
    if isinstance(class2label, str):
        class2label = mv.load_c2l(class2label)
    label2class = {v: k for k, v in class2label.items()}

    if isinstance(dsmd, str):
        dsmd = mv.load_dsmd(dsmd, class2label, mode='det')

    dcm_files = mv.glob(dcm_dir, "*.dcm")
    assert len(dcm_files) == len(dsmd)
    dcm_infos = {}
    for dcm_file in dcm_files:
        ds = mv.dcminfo(dcm_file)
        dcm_infos[mv.filetitle(dcm_file)] = [dcm_file, ds.Rows, ds.Columns]

    for key in dsmd.keys():
        shapes = []
        for i, bboxes in enumerate(dsmd[key]):
            for bbox in bboxes:
                if len(bbox) == 5 and bbox[-1] < thr:
                    continue
                label = label2class[i] if len(bbox) == 4 else \
                    label2class[i] + "_{:.3f}".format(bbox[-1])
                shape = _gen_rws_shape(bbox[:4], label)
                shapes.append(shape)
        if len(shapes) == 0:
            continue
        data = dict(
            version='0.1.0',
            flags={},
            shapes=shapes,
            lineColor=None,
            fillColor=None,
            imagePath=mv.basename(dcm_infos[key][0]),
            imageData=None,
            imageHeight=dcm_infos[key][1],
            imageWidth=dcm_infos[key][2],
        )
        rws_path = mv.joinpath(dcm_dir, key + suffix)
        with open(rws_path, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    dsmd = rws2dsmd_bbox(['/home/huiying/Test/test.json'], 1, lambda x: 0)
    print(dsmd)
