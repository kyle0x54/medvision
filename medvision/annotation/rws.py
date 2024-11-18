import json

import medvision as mv


def get_rws_datainfo_path(dcm_path):
    return mv.change_suffix(dcm_path, ".datainfo")


def get_rws_annot_path(dcm_path):
    return mv.change_suffix(dcm_path, ".json")


def get_rws_flag_path(dcm_path):
    return mv.change_suffix(dcm_path, ".flag")


def get_rws_text_path(dcm_path):
    return mv.change_suffix(dcm_path, ".text")


def _load_rws_contour_single_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # relative path from label file to relative path from cwd
    image_path = mv.joinpath(mv.parentdir(filepath), data["imagePath"])
    height = data.get("imageHeight")
    width = data.get("imageWidth")
    shapes = [{"category": s["label"], "points": s["points"]} for s in data["shapes"]]

    return {
        "height": height,
        "width": width,
        "image_path": image_path,
        "shapes": shapes,
    }


def load_rws_contour(paths):
    """Load rws file (in general contour format).

    Args:
        paths (str or list): rws file path(s).

    Returns:
        rws object (dict), e.g.
        {
            'height': 256,
            'width': 245,
            'image_path': 'example.dcm',
            'shapes':
            [
                {
                    'category': 'cat',
                    'points': [[12, 12], [34, 34], ...]
                },
                {
                    'category': 'dog',
                    'points': [[34, 34], [56, 56], ...]
                },
            ]
        }
    """
    if not isinstance(paths, (tuple, list)):
        return _load_rws_contour_single_file(paths)
    else:
        rws_objs = [_load_rws_contour_single_file(path) for path in paths]
        assert len(rws_objs) > 0, "Do not allow empty rws file path list"
        result = rws_objs[0].copy()
        for rws_obj in rws_objs[1:]:
            for key in ["image_path", "height", "width"]:
                assert (
                    rws_objs[0][key] == rws_obj[key]
                ), "rws files with inconsistent attribute %s [%s != %s]" % (
                    key,
                    rws_objs[0][key],
                    rws_obj[key],
                )
            result["shapes"].extend(rws_obj["shapes"])
        return result


def load_rws_bbox(paths):
    """Load rws file (in bounding box format).

    Args:
        paths (str or list): rws file path(s).

    Returns:
        rws object (dict), e.g.
        {
            'height': 256,
            'width': 245,
            'image_path': 'example.dcm',
            'shapes':
            [
                {
                    'category': 'cat',
                    'bbox': [12, 12, 34, 34]
                },
                {
                    'category': 'dog',
                    'bbox': [34, 34, 56, 56]
                },
            ]
        }

    N.B. this function only support rectangle annotation.
    """
    rws = load_rws_contour(paths)

    shapes = []
    for instance in rws["shapes"]:
        category, points = instance["category"], instance["points"]
        assert len(points) == 2, "only support rectangle annotation"
        xmin = min(points[0][0], points[1][0])
        ymin = min(points[0][1], points[1][1])
        xmax = max(points[0][0], points[1][0])
        ymax = max(points[0][1], points[1][1])
        shape = {"category": category, "bbox": [xmin, ymin, xmax, ymax]}
        shapes.append(shape)

    rws["shapes"] = shapes

    return rws


def _gen_rws_shape_bbox(bbox, label):
    flags = {}
    shape_type = "rectangle"
    line_color = (0, 255, 0, 128)
    fill_color = (255, 0, 0, 128)
    return dict(
        label=label,
        line_color=line_color,
        fill_color=fill_color,
        points=[[float(bbox[0]), float(bbox[1])], [float(bbox[2]), float(bbox[3])]],
        shape_type=shape_type,
        flags=flags,
    )


def save_rws_bbox(
    filepath, shapes, image_shape, suffix=".json", score_thresh=0, fixed_label="auto"
):
    """Save bounding boxes into rws file.

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
        fixed_label (str): bounding box label (for 1 class object detection).
            if "shapes" is a numpy.ndarray, or box list, this label is used.
    """
    assert len(shapes) != 0, "no bounding box for [%s]" % filepath
    assert suffix.startswith(".json"), "unsupported suffix %s" % suffix

    # if not default rws suffix, change suffix
    if suffix != ".json":
        filepath = mv.change_suffix(filepath, suffix)

    # for raw bounding boxes, convert to rws "shape" field format first.
    bboxes = shapes
    shapes = []
    for bbox in bboxes:
        if isinstance(bbox, dict):  # rws "shape" field format
            label, bbox = bbox["category"], bbox["bbox"]
        elif len(bbox) == 5 and bbox[-1] < score_thresh:
            continue
        else:  # ndarray or box list
            label, bbox = fixed_label, bbox[:4]
        shape = _gen_rws_shape_bbox(bbox, label)
        shapes.append(shape)

    # make rws file content
    data = dict(
        version="0.1.0",
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
    with open(filepath, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
