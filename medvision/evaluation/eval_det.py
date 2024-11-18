from collections.abc import Iterable
from collections import OrderedDict

import numpy as np
from natsort import natsorted
from sklearn import metrics as skm

from .compute_overlap import compute_overlap


def _compute_ap_voc07(rec, pre):
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(rec >= t) == 0:
            p = 0
        else:
            p = np.max(pre[rec >= t])
        ap = ap + p / 11.0

    return ap


def _compute_ap_voc12(rec, pre):
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], pre, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * precision
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def _compute_froc(fps, rec, begin=0.125, end=2):
    # to average recall of fps within [begin, end]
    index = [i for i, fp in enumerate(list(fps)) if (begin <= fp <= end)]
    if len(index) == 0:
        return None
    froc_auc = np.sum(rec[index]) / float(len(index))
    return froc_auc


def _compute_ap(rec, pre, use_voc07_metric=False):
    """Compute the average precision, given the recall and precision curves.

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


def _standardize(dts, gts):
    """Convert gt/dt from dict to list[list[ndarray]] (if in dict format).
    Args:
        dts (dict or list[list[ndarray]]): detected bounding boxes for
            different labels in a set of images, each bbox is of shape (n, 5).
        gts (dict or list[list[ndarray]]): ground truth bounding boxes
            for different labels in a set of images, each bbox is of
            shape (n, 4).
            gts[img_id][label_id] = bboxes (for a specific label in an image).
    """
    assert len(gts) == len(dts), "dts and gts must have the same length"

    if isinstance(dts, dict) and isinstance(gts, dict):
        assert set(dts.keys()) == set(gts.keys()), "dts and gts must have the same key set"
        dts = [value for _, value in natsorted(dts.items())]
        gts = [value for _, value in natsorted(gts.items())]

    return dts, gts


def eval_det4binarycls(dts, gts, score_thrs):
    """Evaluate a detector's classification capability.

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
        score_thrs (float or Iterable): confidence threshold to determine 
            whether a detection is valid.
    Returns:
        (dict): a dict containing classification metrics TP, FP, TN, FN,
        accuracy, recall, precision.
    """
    if not isinstance(score_thrs, Iterable):
        score_thrs = [score_thrs]

    assert len(gts) == len(dts)
    dts, gts = _standardize(dts, gts)

    # compute detector's classification capability
    results = {"thrs": {}}
    for score_thr in score_thrs:
        tp, fp, tn, fn = 0, 0, 0, 0
        for i in range(len(gts)):
            assert len(gts[i]) == 1, "only support 1-class detection"
            assert len(dts[i]) == 1, "only support 1-class detection"
            gt = gts[i][0]
            dt = dts[i][0]

            if len(dt) != 0:
                has_dt = dt[:, 4].max() > score_thr
            else:
                has_dt = False

            has_gt = len(gt) != 0

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

        result["tp"] = tp
        result["fp"] = fp
        result["tn"] = tn
        result["fn"] = fn

        eps = np.finfo(np.float32).eps
        result["accuracy"] = (tp + tn) / np.maximum(tp + fn + tn + fp, eps)
        result["sensitivity"] = tp / np.maximum(tp + fn, eps)
        result["specificity"] = tn / np.maximum(tn + fp, eps)
        result["recall"] = result["sensitivity"]
        result["precision"] = tp / np.maximum(tp + fp, eps)

        results["thrs"][str(score_thr)] = result

    # create roc curve
    gt_labels = [(len(gt[0]) != 0) for gt in gts]
    dt_scores = [dt[0][:, -1].max() if len(dt[0]) > 0 else 0 for dt in dts]
    results["roc_curve"] = skm.roc_curve(gt_labels, dt_scores)
    results["roc_auc"] = skm.roc_auc_score(gt_labels, dt_scores)

    return results


def eval_det(dts, gts, num_classes=1, iou_thr=0.5):
    """Evaluate a given dataset by comparing DT with GT.

    Args:
        dts (dict or list[list[ndarray]]): detected bounding boxes for
            different labels in a set of images, each bbox is of shape (n, 5).
        gts (dict or list[list[ndarray]]): ground truth bounding boxes
            for different labels in a set of images, each bbox is of
            shape (n, 4).
            gts[img_id][label_id] = bboxes (for a specific label in an image).
        num_classes (int): number of classes to detect.
        iou_thr (float): threshold to determine whether a detection is
            positive or negative.

    Returns:
        (OrderedDict): AP, number of GT bboxes, FROC curve for each label.
    """
    dts, gts = _standardize(dts, gts)
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
            dt = dt[np.argsort(-dt[:, -1])]
            num_anns += len(gt)
            matched_anns = []

            for d in dt:
                scores = np.append(scores, d[-1])

                if len(gt) == 0:
                    fps = np.append(fps, 1)
                    tps = np.append(tps, 0)
                    continue

                overlaps = compute_overlap(
                    np.expand_dims(d, axis=0).astype(np.float32), gt.astype(np.float32)
                )
                matched_ann = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, matched_ann]

                if max_overlap > iou_thr and matched_ann not in matched_anns:
                    fps = np.append(fps, 0)
                    tps = np.append(tps, 1)
                    matched_anns.append(matched_ann)
                else:
                    fps = np.append(fps, 1)
                    tps = np.append(tps, 0)

        # no annotations -> AP for this class is None
        if num_anns == 0:
            results[label] = OrderedDict()
            results[label]["ap"] = None
            results[label]["num_gt_bboxes"] = 0
            results[label]["froc"] = None
            continue

        # sort by score
        indices = np.argsort(-scores)
        fps, tps, scores = fps[indices], tps[indices], scores[indices]

        # compute false positives and true positives at retrieval cutoff
        # of k bboxes (k in [1, n], n is the total number of detected bboxes)
        fps, tps = np.cumsum(fps), np.cumsum(tps)

        # compute recall and precision
        recall = tps / num_anns
        precision = tps / np.maximum(tps + fps, np.finfo(np.float32).eps)

        # compute AP, number of ground truth bboxes and FROC
        results[label] = OrderedDict()
        results[label]["ap"] = _compute_ap(recall, precision)
        results[label]["num_gt_bboxes"] = num_anns
        results[label]["froc"] = fps / num_imgs, recall  # recall = sensitivity
        results[label]["froc_auc"] = _compute_froc(fps / num_imgs, recall)

        # find score of 0.5 fps and 1 fps.
        # TODO: move to project level code/script
        fps_per_image = fps / num_imgs
        fp_range = np.array(range(10)) * 0.1 + 0.1
        results[label]["pr"] = {}
        for i in range(len(scores) - 1):
            for n in fp_range:
                if fps_per_image[i] < n <= fps_per_image[i + 1]:
                    results[label]["pr"]["fp-" + str(n)[:3]] = {
                        "thr": scores[i],
                        "recall": recall[i],
                        "precision": precision[i],
                    }

    return results
