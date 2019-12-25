import math
import PIL.Image
import PIL.ImageDraw
import medvision as mv
import numpy as np
from tqdm import tqdm

from .rws import load_rws_contour


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


def rws2mask(rws_path, mask_path):
    rws_contour = load_rws_contour(rws_path)

    img_shape = (rws_contour['height'], rws_contour['width'])
    mask = np.zeros(img_shape[:2], dtype=bool)
    shapes = rws_contour['shapes']
    for shape in shapes:
        m = shape_to_mask(img_shape, shape['points'])
        mask = np.bitwise_xor(mask, m)
    mask = mask.astype(np.uint8) * 255

    mv.imwrite(mask_path, mask)


def batch_rws2mask(rws_dir, mask_dir, **kwargs):
    """ Convert rws format annotation to mask format.

    Args:
        rws_dir (str): rws annotation files directory.
        mask_dir (str): mask files directory.

    N.B. dicom file title should be exactly the same with mask file title.
    e.g. 123.dcm, 123.png
    """
    mv.mkdirs(mask_dir)
    rws_filenames = mv.listdir(rws_dir)

    file_titles = [mv.splitext(fn)[0] for fn in rws_filenames]

    for file_title in tqdm(file_titles):
        rws_path = mv.joinpath(rws_dir, file_title + '.json')
        mask_path = mv.joinpath(mask_dir, file_title + '.png')
        rws2mask(rws_path, mask_path, **kwargs)


if __name__ == '__main__':
    # TODO: move to unittest
    rws_dir = '/home/huiying/.rws/rws'
    mask_dir = '/home/huiying/.rws/masks'

    batch_rws2mask(rws_dir, mask_dir)
