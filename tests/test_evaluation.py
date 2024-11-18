import numpy as np

import medvision as mv

DATA_DIR = mv.joinpath(mv.parentdir(__file__), 'data')
DSMD_DET_DT = mv.joinpath(DATA_DIR, 'texts', 'dsmd_det_dt.csv')
DSMD_DET_GT = mv.joinpath(DATA_DIR, 'texts', 'dsmd_det_gt.csv')
DSMD_DET_C2L = mv.joinpath(DATA_DIR, 'texts', 'det_classes.csv')


def test_eval_det_a():
    class2label = mv.load_c2l(DSMD_DET_C2L)
    dts = mv.load_dsmd(DSMD_DET_DT, DSMD_DET_C2L, mode='det')
    gts = mv.load_dsmd(DSMD_DET_GT, DSMD_DET_C2L, mode='det')
    det_metric = mv.eval_det(
        dts, gts, num_classes=len(class2label), iou_thr=0.5)
    ap, num_anns = det_metric[0]['ap'], det_metric[0]['num_gt_bboxes']
    assert 0.263 < ap < 0.264
    assert int(num_anns) == 546

    m = mv.eval_det4binarycls(dts, gts, score_thrs=[0.05])
    assert 0.849 < m['thrs']['0.05']['accuracy'] < 0.850


def test_eval_det_b():
    gts = mv.load_dsmd(DSMD_DET_GT, DSMD_DET_C2L, mode='det')
    gts = [value for key, value in gts.items()]
    dts = []
    for img_id, _ in enumerate(gts):
        dts.append([])
        for label_id, _ in enumerate(gts[img_id]):
            dts[img_id].append([])
            num_bboxes = len(gts[img_id][label_id])
            scores = np.ones((num_bboxes, 1), dtype=np.float32)
            dts[img_id][label_id] = np.hstack(
                [gts[img_id][label_id], scores])

    det_metric = mv.eval_det(dts, gts)
    ap, num_anns = det_metric[0]['ap'], det_metric[0]['num_gt_bboxes']
    assert ap > 0.99
    assert int(num_anns) == 546

    m = mv.eval_det4binarycls(dts, gts, score_thrs=[0.05])
    assert m['thrs']['0.05']['accuracy'] > 0.99
