import json
import cv2
import numpy as np


def load_bdc_dr_annot(annot_path, class2label=None):
    """ Load a DR BDC annotation file.

    Args:
        annot_path (str): annotation file path.
        class2label (dict or callable): convert classname to label. If
            it is not dict or callable, the classname will not be converted.

    Returns:
        (list[ndarray]): a list containing all annotation contours.
    """
    with open(annot_path, 'rt', encoding='GBK') as fd:
        lines = fd.readlines()

    contours = []
    for line in lines:
        classname, content = line.split('|')

        if isinstance(class2label, dict):
            label = class2label[classname]
        elif callable(class2label):
            label = class2label(classname)
        else:
            label = classname

        contour = np.array(json.loads(content)).astype(np.int)
        contour = contour[:, :2]
        contour = np.reshape(contour, (-1, 1, 2))
        contours.append((label, contour))

    return contours


def _contour2bbox(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return [x, y, x + w, y + h]


def load_bdc_dr_bbox(annot_path, class2label=None):
    """ Load a 2D BDC annotation file and convert it to a bounding box.

    Args:
        annot_path (str): annotation file path.
        class2label (dict or callable): convert classname to label. If
            it is not dict or callable, the classname will not be converted.

    Returns:
        (list[ndarray]): a list containing converted annotation bboxes.
    """
    bboxes = []

    contours = load_bdc_dr_annot(annot_path, class2label)
    for label, contour in contours:
        bbox = _contour2bbox(contour)
        bboxes.append((label, bbox))

    return bboxes


if __name__ == '__main__':
    # TODO: move to unittest
    image_path = '/mnt/sdb1/tb/internal/tb/002565.dcm'
    annot_path = '/mnt/sdb1/tb/internal/label/002565.txt'
    bboxes = load_bdc_dr_bbox(annot_path, lambda x: 1)
    import medvision as mv
    image = mv.dcmread_dr(image_path)
    image = mv.normalize_grayscale(image) * 255.0
    image = image.astype(np.uint8)
    tmp = [bbox for label, bbox in bboxes]
    tmp = np.array(tmp)
    mv.imshow_bboxes(image, tmp, thickness=4)
    print(bboxes)
