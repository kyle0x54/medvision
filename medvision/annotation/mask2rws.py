from collections import OrderedDict
import os
from glob import glob
import json

import cv2
import numpy as np
from tqdm import tqdm


RWS_SUFFIX = ".json"
MASK_SUFFIX = ".png"
DICOM_SUFFIX = ".dcm"


def _gen_rws_shape(label, contour):
    line_color = (0, 255, 0, 128)
    fill_color = (255, 0, 0, 128)
    flags = {}
    shape_type = 'polygon'

    return dict(
        label=label,
        line_color=line_color,
        fill_color=fill_color,
        points=[(float(p[0][0]), float(p[0][1])) for p in contour],
        shape_type=shape_type,
        flags=flags
    )


def mask2rws(
    mask: np.ndarray,
    label2category:OrderedDict[int, str],
    dcm_filename: str,
) -> dict:
    if mask.max() == 0:
        raise ValueError("Can not convert empty mask to rws!")

    shapes = []
    for label, category in label2category.items():
        bmask = (mask == label).astype(np.uint8)
        contours, _ = cv2.findContours(
            bmask,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            shape = _gen_rws_shape(category, contour)
            shapes.append(shape)

    return dict(
        version='0.1.0',
        flags={},
        shapes=shapes,
        lineColor=None,
        fillColor=None,
        imagePath=dcm_filename,  #  N.B. Yes, I know filename != filepath, but here we need filename
        imageData=None,
        imageHeight=mask.shape[0],
        imageWidth=mask.shape[1],
    )


def mask2rws_wrapper(
    mask_path: str,
    rws_path: str,
    label2category:OrderedDict[int, str],
):
    dcm_filename = os.path.basename(mask_path)[:-len(MASK_SUFFIX)] + DICOM_SUFFIX
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    rws_data = mask2rws(mask, label2category, dcm_filename)
    with open(rws_path, 'w') as f:
        json.dump(rws_data, f, ensure_ascii=False, indent=2)


def batch_mask2rws(mask_dir, rws_dir, label2category):
    mask_paths = glob(os.path.join(mask_dir, "**", "*" + MASK_SUFFIX), recursive=True)
    for mask_path in tqdm(mask_paths):
        # keep folder structure
        relative_path = os.path.relpath(mask_path, mask_dir)
        rws_path = os.path.join(rws_dir, relative_path)[:-len(MASK_SUFFIX)] + RWS_SUFFIX
        os.makedirs(os.path.dirname(rws_path), exist_ok=True)
        mask2rws_wrapper(mask_path, rws_path, label2category)


# TODO: add unit test
if __name__ == '__main__':
    mask_dir = '/mnt/sdb/mg/breast_masks'
    rws_dir = '/mnt/sdb/mg/breast_rws'
    label2category = OrderedDict([(255, "breast")])
    os.makedirs(rws_dir, exist_ok=True)
    batch_mask2rws(mask_dir, rws_dir, label2category)
