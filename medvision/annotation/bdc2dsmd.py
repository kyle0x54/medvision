import numpy as np
from .load_annotation import load_bdc_dr_bbox
import medvision as mv


def bdc2dsmd_det_2d(annot_dir,
                    image_dir=None,
                    class2label=None,
                    ignore_label_name=True,
                    replace_ext=lambda x: x):
    # N.B. annotation file name and image file name should be the same
    num_classes = len(class2label) if class2label is not None else 1

    filenames = mv.listdir(annot_dir)
    empty_bboxes = np.zeros((0, 4), dtype=np.float32)
    dsmd = {replace_ext(filename): [empty_bboxes] * num_classes
            for filename in filenames}
    for filename in filenames:
        annot_filepath = mv.joinpath(annot_dir, filename)
        bboxes = load_bdc_dr_bbox(
            annot_filepath,
            lambda x: 0 if ignore_label_name else class2label
        )
        for label, bbox in bboxes:
            bbox = np.array(bbox, dtype=np.float32).reshape(-1, 4)
            if dsmd[replace_ext(filename)][label].shape[0] == 0:
                dsmd[replace_ext(filename)][label] = bbox
            else:
                dsmd[replace_ext(filename)][label] = np.append(
                    dsmd[replace_ext(filename)][label], bbox, axis=0)

    return mv.make_dsmd(dsmd)


if __name__ == '__main__':
    # TODO: move to unittest
    annot_dir = '/mnt/sdb1/tb/internal/label'
    dsmd_path = '/mnt/sdb1/tb/internal/1.csv'

    def replace_ext(x):
        return x.replace('.txt', '.dcm')

    dsmd = bdc2dsmd_det_2d(annot_dir, replace_ext=replace_ext)
    mv.save_dsmd(dsmd_path, dsmd, {'tb': 0}, mode='det')

    dsmd = mv.load_dsmd(dsmd_path, {'tb': 0}, mode='det')
    bboxes = dsmd['002565.dcm'][0]
    image_path = '/mnt/sdb1/tb/internal/tb/002565.dcm'
    image = mv.dcmread_dr(image_path)
    image = mv.normalize_grayscale(image) * 255.0
    image = image.astype(np.uint8)
    mv.imshow_bboxes(image, bboxes, thickness=4)
