import ast
import numpy as np
import medvision as mv


def _parse_line(line, class2label):
    """ Parse a single line in detection dsmd.

    The single line content can be
    'data/1.png,170,146,397,681,cat'
    or
    'data/1.png,170,146,397,681,0.5,cat'
    or
    'data/1.png,,,,,'
    Note it can not be 'data/1.png'!

    Args:
        line (str): a single line content in detection dsmd.
        class2label (dict): the dict mapping class to label.
    Return:
        key (str): filename or filepath of an image.
        value (str): ground truth or detection result.
        label (int): the target label value.
    """
    key, value = line.strip().split(',', 1)
    value, class_name = value.strip().rsplit(',', 1)

    if class_name == '':
        return key, None, None
    else:
        value = ast.literal_eval(value.strip())
        return key, value, class2label[class_name]


def _update_data(data, line, class2label):
    """ Insert a single record into dataset metadata

    Args:
        data (dict): refer to 'load_det_dsmd' return value.
        line (str): a single line content in detection dsmd.
        class2label (dict): the dict mapping class to label.
    """
    num_classes = len(class2label)

    key, value, label = _parse_line(line, class2label)

    if label is None:
        data[key] = [None] * num_classes
        return

    if key not in data:
        data[key] = [None] * num_classes
        data[key][label] = [value]
    elif data[key][label] is None:
        data[key][label] = [value]
    else:
        data[key][label].append(value)


def _convert_bboxes_format(data):
    """ Convert bboxes from list of lists to ndarray of shape (n, 4).
    """
    assert len(data) != 0
    num_classes = len(next(iter(data.values())))

    # convert bboxes from list of lists to ndarray
    empty_bboxes = np.zeros((0, 4), dtype=np.float32)
    for key in data:
        for j in range(num_classes):
            if data[key][j] is not None:
                data[key][j] = np.array(data[key][j], dtype=np.float32)
            else:
                data[key][j] = empty_bboxes
    return data


def load_det_dsmd(dsmd_path, c2l_path):
    """ load detection dataset metadata.

    Args:
        dsmd_path (str): dataset metadata file path.
        c2l_path (str): class-to-label file.

    Return:
        (OrderedDict): Loaded dsmd is a OrderedDict looks like
        {
            data/1.png: [
                [bboxes of category 'cat' in ndarray of shape (n, 4)],
                [bboxes of category 'dog' in ndarray of shape (n, 4)],
                ...
            ]
            data/2.png: [
                ...
            ]
            ...
        }
    """
    data = {}
    class2label = mv.load_c2l(c2l_path)

    # label should start from 0
    assert min(class2label.values()) == 0

    with open(dsmd_path, 'r') as fd:
        for line in fd:
            _update_data(data, line, class2label)

    # convert bboxes from list of lists to ndarray
    data = _convert_bboxes_format(data)

    return mv.make_dsmd(data)


def _write_record(fd, key, value, label2class):
    """ Write a single record to dsmd file.

    Args:
        fd (file stream object): file stream.
        key (str): filename or filepath of an image.
        value (str): ground truth or detection result.
        label2class (dict): the dict mapping label to class name.
    """
    if sum([len(i) for i in value]) == 0:
        fd.write('%s,,,,,\n' % str(key))
        return
    else:
        for label, bboxes in enumerate(value):
            class_name = label2class[label]
            for bbox in bboxes:
                bbox = ','.join([str(elem) for elem in bbox])
                line = '%s,%s,%s\n' % (str(key), str(bbox), class_name)
                fd.write(line)


def save_det_dsmd(dsmd_path, data, c2l_path, auto_mkdirs=True):
    """ Save dataset metadata to specified file.

    Args:
        dsmd_path (str): file path to save dataset metadata.
        data (dict): dataset metadata, refer to 'load_dsmd'.
        c2l_path (str): class-to-label file.
        auto_mkdirs (bool): If the parent folder of `file_path` does
            not exist, whether to create it automatically.
    """
    if auto_mkdirs:
        mv.mkdirs(mv.parentdir(dsmd_path))

    # get label->class mapping
    class2label = mv.load_c2l(c2l_path)
    label2class = {value: key for key, value in class2label.items()}

    # write dataset metadata loop
    dsmd = mv.make_dsmd(data)
    with open(dsmd_path, 'w') as fd:
        for key, value in dsmd.items():
            _write_record(fd, key, value, label2class)


# TODO: move to unit test
if __name__ == '__main__':
    file_path = '/home/kyle/Desktop/test/dsmd_test/train_with_invert.csv'
    out_path = '/home/kyle/Desktop/test/dsmd_test/out.csv'
    c2l_path = '/home/kyle/Desktop/test/dsmd_test/c2l.csv'
    dsmd = load_det_dsmd(file_path, c2l_path)

    save_det_dsmd(out_path, dsmd, c2l_path, auto_mkdirs=True)

    dsmd = load_det_dsmd(out_path, c2l_path)
    dsmd = [value for key, value in dsmd.items()]
    dsmd_det = []
    for img_id, _ in enumerate(dsmd):
        dsmd_det.append([])
        for label_id, _ in enumerate(dsmd[img_id]):
            dsmd_det[img_id].append([])
            num_bboxes = len(dsmd[img_id][label_id])
            scores = np.ones((num_bboxes, 1), dtype=np.float32)
            dsmd_det[img_id][label_id] = np.hstack([dsmd[img_id][label_id],
                                                    scores])

    det_metric = mv.eval_det(dsmd_det, dsmd)
    print(det_metric)
