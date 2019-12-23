import json
import medvision as mv


def get_rws_datainfo_path(dcm_path):
    return mv.splitext(dcm_path)[0] + ".datainfo"


def get_rws_annot_path(dcm_path):
    return mv.splitext(dcm_path)[0] + ".json"


def get_rws_flag_path(dcm_path):
    return mv.splitext(dcm_path)[0] + ".flag"


def get_rws_text_path(dcm_path):
    return mv.splitext(dcm_path)[0] + ".text"


def load_rws_contour(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # relative path from label file to relative path from cwd
    image_path = mv.joinpath(mv.parentdir(filepath), data['imagePath'])
    height = data.get('imageHeight')
    width = data.get('imageWidth')
    shapes = []
    for s in data['shapes']:
        shape = (
            s['label'],
            s['points'],
        )
        shapes.append(shape)

    return {
        'height': height,
        'width': width,
        'image_path': image_path,
        'shapes': shapes
    }


def load_rws_bbox(filepath):
    rws_bboxes = load_rws_contour(filepath)

    shapes = []
    for label, points in rws_bboxes['shapes']:
        assert len(points) == 2, 'only support rectangle annotation'
        xmin = min(points[0][0], points[1][0])
        ymin = min(points[0][1], points[1][1])
        xmax = max(points[0][0], points[1][0])
        ymax = max(points[0][1], points[1][1])
        shape = (
            label,
            [xmin, ymin, xmax, ymax]
        )
        shapes.append(shape)

    rws_bboxes['shapes'] = shapes

    return rws_bboxes


def gen_rws_shape_bbox(bbox, label):
    flags = {}
    shape_type = 'rectangle'
    return dict(
        label=label,
        points=[[float(bbox[0]), float(bbox[1])],
                [float(bbox[2]), float(bbox[3])]],
        shape_type=shape_type,
        flags=flags
    )


def save_rws_bbox(
    filepath,
    shapes,
    image_shape,
    suffix=".json",
    score_thresh=0,
    label_="auto"
):
    """
    Save bounding boxes into rws file.

    Args:
        filepath (str): rws file path.
        shapes (list or numpy.ndarray): "shape" field in rws file or
            raw bounding boxes (in numpy.ndarray format or list format,
            in this case, only single-class label is supported).
        image_shape (tuple): shape of corresponding image, in (h, w) format.
        suffix (str): rws file suffix, e.g. ".json", ".json_EA", ".json_A1".
        score_thresh (float): probability threshold of bounding boxes.
            If probability score of a box is larger than the threshold,
            the box will be saved into rws file, otherwise not.
        label_ (str): bounding box label (for single class object detection).
    """
    assert len(shapes) != 0, "no bounding box for [%s]" % filepath
    assert suffix.startswith(".json"), "unsupported suffix %s" % suffix

    # if not default rws suffix, change suffix
    if suffix != ".json":
        filepath = mv.joinpath(
            mv.parentdir(filepath),
            mv.filetitle(filepath) + suffix
        )

    # for raw bounding boxes, convert to rws "shape" field format first.
    bboxes = shapes
    shapes = []
    for bbox in bboxes:
        if len(bbox) == 2:  # rws "shape" field format
            label, bbox = bbox
        elif len(bbox) == 5 and bbox[-1] < score_thresh:
            continue
        else:  # ndarray or box list
            label, bbox = label_, bbox[:4]
        shape = gen_rws_shape_bbox(bbox, label)
        shapes.append(shape)

    # establish rws file content
    data = dict(
        version='0.1.0',
        flags={},
        shapes=shapes,
        lineColor=None,
        fillColor=None,
        imagePath=mv.filetitle(filepath) + ".dcm",
        imageData=None,
        imageHeight=image_shape[0],
        imageWidth=image_shape[1],
    )

    # save rws file
    with open(filepath, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    rws_filepath = '0.json'
    bboxes = load_rws_bbox(rws_filepath)
    print(bboxes)
    save_rws_bbox(
        mv.joinpath(
            mv.parentdir(rws_filepath),
            mv.filetitle(rws_filepath) + '.json_A1'
        ),
        bboxes['shapes'],
        (2947, 3000),
        suffix='.json_A1',
        label_='saved_box'
    )
