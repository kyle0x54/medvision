import numpy as np
from compute_overlap import compute_overlap


def compute_ap(rec, pre, use_voc07_metric=False):
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
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(pre[rec >= t])
            ap = ap + p / 11.
    else:
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


def eval_det(gts, dts, num_classes=1, iou_thr=0.5, score_thr=0.05,
             num_max_det=100, save_path=None):
    """ Evaluate a given dataset using a given model.

    Args:
        gts (list[list[ndarray]]): ground truth bounding boxes for
            different labels in a set of images, each bbox is of the
            shape (n, 4).
            gts[img_id][label_id] = bboxes (for a specific label in an image).
        dts (list[list[ndarray]]): detected bounding boxes for
            different labels in a set of images, each bbox is of the
            shape (n, 5).
        num_classes (int): number of classes to detect.
        iou_thr (float): threshold to determine whether a detection is
            positive or negative.
        score_thr : score confidence threshold to determine whether a
            detection is valid.
        num_max_det (int): maximum number of detections to use per image.
        save_path (str): path to save images with detection results.

    Returns
        (dict): a dict mapping class names to mAP scores.
    """
    assert len(gts) == len(dts)
    num_imgs = len(gts)

    average_precisions = {}

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
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        fps = fps[indices]
        tps = tps[indices]

        # compute false positives and true positives at retrieval cutoff
        # of k bboxes (k in [1, n], n is the total number of detected bboxes)
        fps = np.cumsum(fps)
        tps = np.cumsum(tps)

        # compute recall and precision
        recall = tps / num_anns
        precision = tps / np.maximum(tps + fps, np.finfo(np.float32).eps)

        # compute AP
        average_precision = compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_anns

    return average_precisions
