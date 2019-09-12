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


if __name__ == '__main__':
    dsmd = rws2dsmd_bbox(['/home/huiying/Test/test.json'], 1, lambda x: 0)
    print(dsmd)
