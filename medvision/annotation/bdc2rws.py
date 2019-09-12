import logging
import json
import medvision as mv
from tqdm import tqdm
from .bdc import load_bdc_dr_contour


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


def bdc2rws_contour(dcm_path, bdc_path, rws_path):
    ds = mv.dcminfo(dcm_path)
    contours = load_bdc_dr_contour(bdc_path)

    shapes = []
    for label, contour in contours:
        shape = _gen_rws_shape(label, contour)
        shapes.append(shape)

    data = dict(
        version='0.1.0',
        flags={},
        shapes=shapes,
        lineColor=None,
        fillColor=None,
        imagePath=mv.basename(dcm_path),
        imageData=None,
        imageHeight=ds.Rows,
        imageWidth=ds.Columns,
    )

    with open(rws_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def batch_bdc2rws_contour(dcm_dir, bdc_dir, rws_dir, **kwargs):
    """ Convert BDC format annotation to rws format.

    Args:
        dcm_dir (str): dicom files directory.
        bdc_dir (str): bdc annotation files directory.
        rws_dir (str): rws annotation files directory.

    N.B.
        dicom title should be exactly the same with annotation
        file title. e.g. 123.dcm, 123.txt
    """
    mv.mkdirs(rws_dir)
    dcm_filenames = mv.listdir(dcm_dir)
    bdc_filenames = mv.listdir(bdc_dir)

    dcm_titles = [mv.splitext(fn)[0] for fn in dcm_filenames]
    bdc_titles = [mv.splitext(fn)[0] for fn in bdc_filenames]
    file_titles = list(set(dcm_titles).intersection(set(bdc_titles)))

    if (len(dcm_filenames) != len(bdc_filenames) or
            len(file_titles) != len(dcm_filenames)):
        logging.warning('dicoms & annotations do not exactly match')

    for file_title in tqdm(file_titles):
        dcm_path = mv.joinpath(dcm_dir, file_title + '.dcm')
        bdc_path = mv.joinpath(bdc_dir, file_title + '.txt')
        rws_path = mv.joinpath(rws_dir, file_title + '.json')
        bdc2rws_contour(dcm_path, bdc_path, rws_path, **kwargs)


if __name__ == '__main__':
    # TODO: move to unittest
    dcm_dir = '/mnt/sdd1/Backup/TB_wangyu/Train/sub_image'
    bdc_dir = '/mnt/sdd1/Backup/TB_wangyu/Train/sub_label'
    rws_dir = '/mnt/sdd1/Backup/TB_wangyu/Train/rws'

    batch_bdc2rws_contour(dcm_dir, bdc_dir, rws_dir)
