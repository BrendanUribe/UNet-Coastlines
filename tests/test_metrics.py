"""Tests for the evaluation metrics."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import metrics as M  # noqa: E402


def test_confusion_and_iou_perfect():
    t = np.array([[0, 1], [2, 2]])
    cm = M.confusion_matrix(t, t, 3)
    assert np.allclose(np.diag(cm), [1, 1, 2])
    assert np.allclose(M.iou_from_confusion(cm), 1.0)
    assert M.pixel_accuracy_from_confusion(cm) == 1.0


def test_iou_absent_class_is_nan_not_zero():
    """A class present in neither prediction nor truth must not drag mIoU down."""
    t = np.zeros((4, 4), int)
    cm = M.confusion_matrix(t, t, 3)
    iou = M.iou_from_confusion(cm)
    assert iou[0] == 1.0
    assert np.isnan(iou[1]) and np.isnan(iou[2])
    assert np.nanmean(iou) == 1.0


def test_tolerance_matching_forgives_one_pixel_shift():
    """The point of tolerance matching: a 1 px offset is not a total miss."""
    gt = np.zeros((32, 32), bool)
    gt[16, 4:28] = True
    shifted = np.roll(gt, 1, axis=0)

    tp, fp, fn = M.match_counts(shifted, gt, tolerance=0.0)
    assert M.prf_from_counts(tp, fp, fn)[2] == 0.0

    tp, fp, fn = M.match_counts(shifted, gt, tolerance=2.0)
    assert M.prf_from_counts(tp, fp, fn)[2] > 0.99


def test_match_counts_empty_cases():
    gt = np.zeros((8, 8), bool)
    gt[4, 2:6] = True
    assert M.match_counts(np.zeros_like(gt), gt) == (0, 0, int(gt.sum()))
    assert M.match_counts(gt, np.zeros_like(gt)) == (0, int(gt.sum()), 0)


def test_boundary_iou_penalises_thickening():
    """Boundary IoU should notice a fattened prediction that plain IoU on a
    thin line barely distinguishes."""
    gt = np.zeros((40, 40), bool)
    gt[20, 5:35] = True
    fat = gt.copy()
    fat[19, 5:35] = True
    fat[21, 5:35] = True
    assert M.boundary_iou(gt, gt) == 1.0
    assert M.boundary_iou(fat, gt) < 1.0


def test_hausdorff_matches_known_offset():
    gt = np.zeros((32, 32), bool)
    gt[10, 5:25] = True
    pred = np.roll(gt, 3, axis=0)
    assert M.hausdorff_percentile(pred, gt, 95.0) == pytest.approx(3.0, abs=0.5)
    assert np.isnan(M.hausdorff_percentile(np.zeros_like(gt), gt))


def test_edge_metrics_perfect_and_empty():
    gt = np.zeros((32, 32), bool)
    gt[16, 4:28] = True

    em = M.EdgeMetrics(thresholds=np.array([0.5]))
    em.update(gt.astype(np.float32), gt)
    r = em.compute()
    assert r["ODS_F1"] == pytest.approx(1.0)
    assert r["BoundaryIoU"] == pytest.approx(1.0)
    assert r["HD95_px"] == pytest.approx(0.0)

    em2 = M.EdgeMetrics(thresholds=np.array([0.5]))
    em2.update(np.zeros((32, 32), np.float32), gt)
    assert em2.compute()["ODS_F1"] == 0.0


def test_edge_metrics_respects_valid_mask():
    """Predictions in unsupervised regions must not count as false positives."""
    gt = np.zeros((32, 32), bool)
    gt[16, 4:16] = True
    valid = np.zeros((32, 32), bool)
    valid[:, :16] = True

    prob = gt.astype(np.float32).copy()
    prob[16, 20:28] = 1.0          # confident garbage, outside the valid region

    em = M.EdgeMetrics(thresholds=np.array([0.5]))
    em.update(prob, gt, valid)
    assert em.compute()["ODS_F1"] == pytest.approx(1.0)


def test_seg_metrics_accumulates():
    sm = M.SegMetrics(3, {0: "a", 1: "b", 2: "c"})
    t = np.array([[0, 1], [2, 2]])
    sm.update(t, t)
    sm.update(t, t)
    r = sm.compute()
    assert r["pixel_acc"] == 1.0 and r["mIoU"] == 1.0
    assert r["IoU/a"] == 1.0


def test_navigation_conversions():
    """Pin the numbers quoted in the review, so they stay honest."""
    # 35 mm lens, 4.96 um pixels -> f = 7056 px -> 1 px ~ 29 arcsec
    assert M.pixels_to_angle_arcsec(1.0, 35.0, 4.96e-3) == pytest.approx(29.2, abs=0.5)
    # 1 px of limb radius at 75,000 km, native 602 px radius -> ~125 km
    assert M.limb_error_to_range_error_km(1.0, 602.0, 75000.0) == pytest.approx(124.6, abs=1.0)
    # the same error after a 256x256 downsample (75 px radius) -> ~1000 km
    assert M.limb_error_to_range_error_km(1.0, 75.0, 75000.0) == pytest.approx(1000.0, abs=10.0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
