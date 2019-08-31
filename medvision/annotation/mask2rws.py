import json
import cv2
import medvision as mv
from tqdm import tqdm


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


def mask2rws(mask_path, rws_path, dcm_path=None):
    if isinstance(mask_path, str):
        mask = mv.imread(mask_path, mv.ImreadMode.GRAY)
        dcm_path = mv.basename(mask_path).replace('.png', '.dcm')
    else:
        mask = mask_path
        assert dcm_path is not None

    if mask.max() == 0:
        print('skip mask with no annotation regions: {}'.format(mask_path))
        return

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    shapes = []
    for contour in contours:
        shape = _gen_rws_shape('auto', contour)
        shapes.append(shape)

    data = dict(
        version='0.1.0',
        flags={},
        shapes=shapes,
        lineColor=None,
        fillColor=None,
        imagePath=dcm_path,
        imageData=None,
        imageHeight=mask.shape[0],
        imageWidth=mask.shape[1],
    )

    with open(rws_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def batch_mask2rws(mask_dir, rws_dir, **kwargs):
    """ Convert mask format annotation to rws format.

    Args:
        mask_dir (str): mask files directory.
        rws_dir (str): rws annotation files directory.

    N.B. dicom file title should be exactly the same with mask file title.
    e.g. 123.dcm, 123.png
    """
    mv.mkdirs(rws_dir)
    mask_filenames = mv.listdir(mask_dir)

    file_titles = [mv.splitext(fn)[0] for fn in mask_filenames]

    for file_title in tqdm(file_titles):
        mask_path = mv.joinpath(mask_dir, file_title + '.png')
        rws_path = mv.joinpath(rws_dir, file_title + '.json')
        mask2rws(mask_path, rws_path, **kwargs)


if __name__ == '__main__':
    # TODO: move to unittest
    mask_dir = '/home/huiying/.rws/masks'
    rws_dir = '/home/huiying/.rws/rws_verification'

    batch_mask2rws(mask_dir, rws_dir)
