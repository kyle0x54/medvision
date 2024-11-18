from collections import OrderedDict
from glob import glob
import os
import math
import uuid

import cv2
import PIL.Image
import PIL.ImageDraw
import numpy as np
from tqdm import tqdm

from .rws import load_rws_contour


RWS_SUFFIX = ".json"
MASK_SUFFIX = ".png"


def shape_to_mask(img_shape, points, shape_type=None,
                  line_width=10, point_size=5):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    if shape_type == 'circle':
        assert len(xy) == 2, 'Shape of shape_type=circle must have 2 points'
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
    elif shape_type == 'rectangle':
        assert len(xy) == 2, 'Shape of shape_type=rectangle must have 2 points'
        draw.rectangle(xy, outline=1, fill=1)
    elif shape_type == 'line':
        assert len(xy) == 2, 'Shape of shape_type=line must have 2 points'
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == 'linestrip':
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == 'point':
        assert len(xy) == 1, 'Shape of shape_type=point must have 1 points'
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    else:
        if len(xy) > 2:
            draw.polygon(xy=xy, outline=1, fill=1)
        else:
            pass
    mask = np.array(mask, dtype=bool)
    return mask


def rws2mask_single_category(rws_contour, label_value=255):
    # N.B. all categories are merged into a single label
    img_shape = (rws_contour['height'], rws_contour['width'])
    mask = np.zeros(img_shape[:2], dtype=bool)
    shapes = rws_contour['shapes']
    for shape in shapes:
        m = shape_to_mask(img_shape, shape['points'])
        mask = np.bitwise_xor(mask, m)  # TODO: bitwise_or?
    mask = mask.astype(np.uint8) * label_value

    return mask


def rws2mask(rws_contour, label_name_to_value):
    img_shape = (rws_contour['height'], rws_contour['width'])
    shapes = rws_contour['shapes']

    cls = np.zeros(img_shape[:2], dtype=np.int32)
    ins = np.zeros_like(cls)
    instances = []
    for shape in shapes:
        points = shape["points"]
        cls_name = shape["category"]
        group_id = uuid.uuid1()
        shape_type = shape.get("shape_type", None)

        instance = (cls_name, group_id)

        if instance not in instances:
            instances.append(instance)
        ins_id = instances.index(instance) + 1
        cls_id = label_name_to_value[cls_name]

        mask = shape_to_mask(img_shape[:2], points, shape_type)
        cls[mask] = cls_id
        ins[mask] = ins_id

    return cls, ins


def rws2multimasks(
    rws_contour,
    label_name_to_value: dict[str, int],
) -> list[np.ndarray]:
    # N.B. This function is used to handle overlapping contour annotations
    img_shape = (rws_contour['height'], rws_contour['width'])
    shapes = rws_contour['shapes']

    result = []
    for category, label in label_name_to_value.items():
        mask = np.zeros(img_shape[:2], dtype=bool)
        for shape in shapes:
            if shape['category'] != category:
                continue
            m = shape_to_mask(img_shape, shape['points'])
            mask = np.bitwise_xor(mask, m)
        result.append(mask.astype(np.uint8) * label)

    return result


def rws2mask_wrapper(
    rws_path: str,
    mask_path: str,
    category2label: OrderedDict[str, int]
):
    rws_contour = load_rws_contour(rws_path)
    mask, _ = rws2mask(rws_contour, category2label)
    cv2.imwrite(mask_path, mask)


def batch_rws2mask(rws_dir, mask_dir, category2label):
    """Convert RWS format annotations to mask format.

    Args:
        rws_dir (str): Directory containing RWS annotation files.
        mask_dir (str): Directory to save the output mask files.

    Note:
        The DICOM file name must match the mask file name exactly.
        For example: 123.dcm should correspond to 123.png.
    """
    rws_paths = glob(os.path.join(rws_dir, "**", "*" + RWS_SUFFIX), recursive=True)
    for rws_path in tqdm(rws_paths):
        # keep folder structure
        relative_path = os.path.relpath(rws_path, rws_dir)
        mask_path = os.path.join(mask_dir, relative_path)[:-len(RWS_SUFFIX)] + MASK_SUFFIX
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        rws2mask_wrapper(rws_path, mask_path, category2label)


# TODO: add unit test
if __name__ == '__main__':
    rws_dir = '/mnt/sdb/mg/breast_rws'
    mask_dir = '/mnt/sdb/mg/breast_rws_masks'
    category2label = OrderedDict([('breast', 128), ('muscle', 255)])
    os.makedirs(mask_dir, exist_ok=True)
    batch_rws2mask(rws_dir, mask_dir, category2label)
