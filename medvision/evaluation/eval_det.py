from collections import OrderedDict
import numpy as np
from .compute_overlap import compute_overlap


def _compute_ap_voc07(rec, pre):
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(rec >= t) == 0:
            p = 0
        else:
            p = np.max(pre[rec >= t])
        ap = ap + p / 11.

    return ap


def _compute_ap_voc12(rec, pre):
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], pre, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * precision
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def _compute_ap(rec, pre, use_voc07_metric=False):
    """ Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    Args:
        rec (list): the recall curve .
        pre (list): the precision curve.
        use_voc07_metric (bool): whether to voc2007 11 points metric.

    Returns
        The average precision as computed in py-faster-rcnn.
    """
    if use_voc07_metric:
        return _compute_ap_voc07(rec, pre)
    else:
        return _compute_ap_voc12(rec, pre)


def eval_det4cls(dts, gts, num_classes=1):
    """ Evaluate a detector's classification capability.

    Only support 2-class classification. If (at least) 1 target is detected in
    an image, the image is considered to be 'positive'. Otherwise, the image
    is considered to be 'negative'.

    Args:
        dts (OrderedDict or list[list[ndarray]]): detected bounding boxes for
            different labels in a set of images, each bbox is of shape (n, 5).
        gts (OrderedDict or list[list[ndarray]]): ground truth bounding boxes
            for different labels in a set of images, each bbox is of
            shape (n, 4).
            gts[img_id][label_id] = bboxes (for a specific label in an image).
        num_classes (int): number of classes to detect.

    Returns
        (dict): a dict containing classification metrics TP, FP, TN, FN,
        accuracy, recall, precision.
    """
    assert len(gts) == len(dts)
    assert num_classes == 1

    # convert gt/dt from dict to list
    if isinstance(dts, dict) and isinstance(gts, dict):
        dts = [value for key, value in dts.items()]
        gts = [value for key, value in gts.items()]

    # compute detector's classification capability
    tp, fp, tn, fn = 0, 0, 0, 0
    for i in range(len(gts)):
        gt = gts[i][0]
        dt = dts[i][0]

        if dt.any() and gt.any():
            tp += 1
        elif dt.any() and not gt.any():
            fp += 1
        elif not dt.any() and gt.any():
            fn += 1
        else:  # not dt.any() and not gt.any():
            tn += 1

    # build result
    result = OrderedDict()
    result['tp'] = tp
    result['fp'] = fp
    result['tn'] = tn
    result['fn'] = fn
    result['accuracy'] = (tp + tn) / (tp + fn + tn + fp)
    result['recall'] = tp / (tp + fn)
    result['precision'] = tp / np.maximum(tp + fp, np.finfo(np.float32).eps)

    return result


def eval_det(dts, gts, num_classes=1, iou_thr=0.5, score_thr=0.05):
    """ Evaluate a given dataset by comparing DT with GT.

    Args:
        dts (OrderedDict or list[list[ndarray]]): detected bounding boxes for
            different labels in a set of images, each bbox is of shape (n, 5).
        gts (OrderedDict or list[list[ndarray]]): ground truth bounding boxes
            for different labels in a set of images, each bbox is of
            shape (n, 4).
            gts[img_id][label_id] = bboxes (for a specific label in an image).
        num_classes (int): number of classes to detect.
        iou_thr (float): threshold to determine whether a detection is
            positive or negative.
        score_thr : score confidence threshold to determine whether a
            detection is valid.

    Returns
        (dict): map label to (average precision, number of GT bboxes).
        (dict): map label to (average false positive per image, sensitivity).
    """
    assert len(gts) == len(dts)
    num_imgs = len(gts)

    if isinstance(dts, dict) and isinstance(gts, dict):
        dts = [value for key, value in dts.items()]
        gts = [value for key, value in gts.items()]

    aps = {}
    frocs = {}

    # match ground truths and detection results
    for label in range(num_classes):
        fps = np.zeros((0,), np.float32)
        tps = np.zeros((0,), np.float32)
        scores = np.zeros((0,), np.float32)
        num_anns = 0.0

        for i in range(num_imgs):
            gt = gts[i][label]
            dt = dts[i][label]
            num_anns += len(gt)
            matched_anns = []

            for d in dt:
                scores = np.append(scores, d[4])

                if len(gt) == 0:
                    fps = np.append(fps, 1)
                    tps = np.append(tps, 0)
                    continue

                overlaps = compute_overlap(np.expand_dims(d, axis=0), gt)
                matched_ann = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, matched_ann]

                if (max_overlap > iou_thr and
                        matched_ann not in matched_anns):
                    fps = np.append(fps, 0)
                    tps = np.append(tps, 1)
                    matched_anns.append(matched_ann)
                else:
                    fps = np.append(fps, 1)
                    tps = np.append(tps, 0)

        # no annotations -> AP for this class is 0
        if num_anns == 0:
            aps[label] = 0, 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        fps, tps = fps[indices], tps[indices]

        # compute false positives and true positives at retrieval cutoff
        # of k bboxes (k in [1, n], n is the total number of detected bboxes)
        fps, tps = np.cumsum(fps), np.cumsum(tps)

        # compute recall and precision
        recall = tps / num_anns
        precision = tps / np.maximum(tps + fps, np.finfo(np.float32).eps)

        # compute AP
        aps[label] = _compute_ap(recall, precision), num_anns

        # compute FROC
        frocs[label] = fps / num_imgs, recall  # recall = sensitivity

    return aps, frocs
