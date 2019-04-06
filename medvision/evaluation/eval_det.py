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

    Returns:
        The average precision as computed in py-faster-rcnn.
    """
    if use_voc07_metric:
        return _compute_ap_voc07(rec, pre)
    else:
        return _compute_ap_voc12(rec, pre)


def _to_list(dts, gts):
    """ Convert gt/dt from OrderedDict to list[list[ndarray]].
    Args:
        dts (OrderedDict or list[list[ndarray]]): detected bounding boxes for
            different labels in a set of images, each bbox is of shape (n, 5).
        gts (OrderedDict or list[list[ndarray]]): ground truth bounding boxes
            for different labels in a set of images, each bbox is of
            shape (n, 4).
            gts[img_id][label_id] = bboxes (for a specific label in an image).
    """
    if isinstance(dts, OrderedDict):
        dts = [value for key, value in dts.items()]

    if isinstance(gts, OrderedDict):
        gts = [value for key, value in gts.items()]

    return dts, gts


def eval_det4binarycls(dts, gts, score_thr=0.05):
    """ Evaluate a detector's classification capability.

    Only support 1-class detection. If (at least) 1 target is detected in
    an image, the image is considered to be 'positive'. Otherwise, the image
    is considered to be 'negative'.

    Args:
        dts (OrderedDict or list[list[ndarray]]): detected bounding boxes for
            different labels in a set of images, each bbox is of shape (n, 5).
        gts (OrderedDict or list[list[ndarray]]): ground truth bounding boxes
            for different labels in a set of images, each bbox is of
            shape (n, 4).
            gts[img_id][label_id] = bboxes (for a specific label in an image).
        score_thr (float): score confidence threshold to determine whether a
            detection is valid.
    Returns:
        (dict): a dict containing classification metrics TP, FP, TN, FN,
        accuracy, recall, precision.
    """
    assert len(gts) == len(dts)
    dts, gts = _to_list(dts, gts)

    # compute detector's classification capability
    tp, fp, tn, fn = 0, 0, 0, 0
    for i in range(len(gts)):
        assert len(gts[i]) == 1, 'only support 1-class detection'
        assert len(dts[i]) == 1, 'only support 1-class detection'
        gt = gts[i][0]
        dt = dts[i][0]

        if len(dt) != 0:
            has_dt = dt[:, 4].max() > score_thr
        else:
            has_dt = False

        has_gt = (len(gt) != 0)

        if has_dt and has_gt:
            tp += 1
        elif has_dt and not has_gt:
            fp += 1
        elif not has_dt and has_gt:
            fn += 1
        else:  # not has_dt and not has_gt:
            tn += 1

    # build result
    result = OrderedDict()

    result['tp'] = tp
    result['fp'] = fp
    result['tn'] = tn
    result['fn'] = fn

    eps = np.finfo(np.float32).eps
    result['accuracy'] = (tp + tn) / np.maximum(tp + fn + tn + fp, eps)
    result['sensitivity'] = tp / np.maximum(tp + fn, eps)
    result['specificity'] = tn / np.maximum(tn + fp, eps)
    result['recall'] = result['sensitivity']
    result['precision'] = tp / np.maximum(tp + fp, eps)

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
        score_thr (float): score confidence threshold to determine whether a
            detection is valid.

    Returns:
        (OrderedDict): AP, number of GT bboxes, FROC curve for each label.
    """
    assert len(gts) == len(dts)
    dts, gts = _to_list(dts, gts)
    num_imgs = len(gts)

    results = OrderedDict()

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

        # no annotations -> AP for this class is None
        if num_anns == 0:
            results[label] = OrderedDict()
            results[label]['ap'] = None
            results[label]['num_gt_bboxes'] = 0
            results[label]['froc'] = None
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

        # compute AP, number of ground truth bboxes and FROC
        results[label] = OrderedDict()
        results[label]['ap'] = _compute_ap(recall, precision)
        results[label]['num_gt_bboxes'] = num_anns
        results[label]['froc'] = fps / num_imgs, recall  # recall = sensitivity

    return results
