"""Evaluation metrics for segmentation and boundary detection.

The original repository computed no metrics at all - ``predict_unet.py``
imported numpy with the comment "math, ioU, Dice" but neither was ever
calculated.  Without a held-out number, no architectural claim is falsifiable.

Plain IoU is reported for completeness but is the wrong headline number for a
one-pixel-wide structure: a thin line has almost no area, so IoU is dominated
by trivial misalignment.  The boundary metrics here are the ones to quote:

* **Boundary IoU** - Cheng, Girshick, Dollar, Kirillov & He, CVPR 2021
  (arXiv:2103.16562).  IoU restricted to a band around the boundary.
* **Tolerance-matched precision/recall/F1** - the BSDS500 edge-detection
  protocol (Arbelaez, Maire, Fowlkes & Malik, IEEE TPAMI 33(5), 2011), used by
  HED and HED-UNet.  ODS is the best F1 at a single threshold across the whole
  set; OIS allows a per-image threshold.
* **95th-percentile Hausdorff distance** - worst-case localisation error in
  pixels, robust to a handful of outliers.  This is the metric that converts
  into a navigation error bound, because a limb or landmark mislocated by *d*
  pixels maps directly onto an angular error of ``d / f_px`` radians.

All functions take numpy arrays and are framework-independent so they can be
used from the training loop or from a standalone evaluation.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


# --------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------
def confusion_matrix(pred, target, num_classes, ignore=None):
    """``(num_classes, num_classes)`` matrix, rows = target, cols = prediction."""
    pred, target = np.asarray(pred).ravel(), np.asarray(target).ravel()
    if ignore is not None:
        keep = ~np.asarray(ignore).ravel()
        pred, target = pred[keep], target[keep]
    k = (target >= 0) & (target < num_classes)
    return np.bincount(
        num_classes * target[k].astype(int) + pred[k].astype(int),
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)


def iou_from_confusion(cm):
    """Per-class IoU; NaN for classes absent from both prediction and target."""
    inter = np.diag(cm).astype(np.float64)
    union = cm.sum(1) + cm.sum(0) - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, inter / union, np.nan)


def dice_from_confusion(cm):
    inter = np.diag(cm).astype(np.float64)
    denom = cm.sum(1) + cm.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, 2 * inter / denom, np.nan)


def pixel_accuracy_from_confusion(cm):
    total = cm.sum()
    return float(np.diag(cm).sum() / total) if total else float("nan")


# --------------------------------------------------------------------------
# Boundary
# --------------------------------------------------------------------------
def boundary_iou(pred_mask, gt_mask, dilation: int = 2):
    """IoU computed only within ``dilation`` pixels of either boundary.

    Cheng et al., CVPR 2021.  ``dilation`` is the band half-width in pixels.
    """
    pred_mask, gt_mask = np.asarray(pred_mask, bool), np.asarray(gt_mask, bool)
    if not pred_mask.any() and not gt_mask.any():
        return float("nan")

    band = (distance_transform_edt(~gt_mask) <= dilation) | (
        distance_transform_edt(~pred_mask) <= dilation
    )
    p, g = pred_mask & band, gt_mask & band
    union = (p | g).sum()
    return float((p & g).sum() / union) if union else float("nan")


def match_counts(pred_mask, gt_mask, tolerance: float = 2.0):
    """Tolerance-matched TP/FP/FN counts for thin structures.

    A predicted pixel counts as correct if a ground-truth pixel lies within
    ``tolerance``, and vice versa.  Without a tolerance, a one-pixel offset
    scores zero, which is why raw pixel F1 is meaningless for boundaries.
    """
    pred_mask, gt_mask = np.asarray(pred_mask, bool), np.asarray(gt_mask, bool)
    n_pred, n_gt = int(pred_mask.sum()), int(gt_mask.sum())
    if n_gt == 0:
        return 0, n_pred, 0
    if n_pred == 0:
        return 0, 0, n_gt

    d_to_gt = distance_transform_edt(~gt_mask)
    d_to_pred = distance_transform_edt(~pred_mask)
    tp = int((d_to_gt[pred_mask] <= tolerance).sum())   # matched predictions
    fp = n_pred - tp
    fn = int((d_to_pred[gt_mask] > tolerance).sum())    # unmatched truth
    return tp, fp, fn


def prf_from_counts(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    if not (precision > 0) or not (recall > 0):
        return precision, recall, 0.0
    return precision, recall, 2 * precision * recall / (precision + recall)


def hausdorff_percentile(pred_mask, gt_mask, percentile: float = 95.0):
    """Symmetric ``percentile``-th Hausdorff distance in pixels.

    Returns NaN when either mask is empty, since the distance is undefined.
    """
    pred_mask, gt_mask = np.asarray(pred_mask, bool), np.asarray(gt_mask, bool)
    if not pred_mask.any() or not gt_mask.any():
        return float("nan")
    d_to_gt = distance_transform_edt(~gt_mask)[pred_mask]
    d_to_pred = distance_transform_edt(~pred_mask)[gt_mask]
    return float(max(np.percentile(d_to_gt, percentile),
                     np.percentile(d_to_pred, percentile)))


# --------------------------------------------------------------------------
# Accumulators
# --------------------------------------------------------------------------
def _nanmean(values):
    """``np.nanmean`` that returns NaN for an all-NaN input without warning."""
    arr = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(arr)) if arr.size and not np.all(np.isnan(arr)) else float("nan")


class SegMetrics:
    """Accumulate a confusion matrix over a dataset."""

    def __init__(self, num_classes: int, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or {i: str(i) for i in range(num_classes)}
        self.cm = np.zeros((num_classes, num_classes), np.int64)

    def update(self, pred, target, ignore=None):
        self.cm += confusion_matrix(pred, target, self.num_classes, ignore)

    def compute(self):
        iou, dice = iou_from_confusion(self.cm), dice_from_confusion(self.cm)
        out = {
            "pixel_acc": pixel_accuracy_from_confusion(self.cm),
            "mIoU": _nanmean(iou),
            "mDice": _nanmean(dice),
        }
        for c in range(self.num_classes):
            out[f"IoU/{self.class_names[c]}"] = float(iou[c])
        return out


class EdgeMetrics:
    """Accumulate boundary metrics for one channel over a dataset.

    ``thresholds`` are swept so ODS (one global threshold) and OIS (best
    per-image threshold) can both be reported, as in the BSDS protocol.
    """

    def __init__(self, thresholds=None, tolerance: float = 2.0, boundary_dilation: int = 2):
        self.thresholds = np.asarray(
            thresholds if thresholds is not None else np.arange(0.05, 1.0, 0.05)
        )
        self.tolerance = tolerance
        self.boundary_dilation = boundary_dilation
        self.counts = np.zeros((len(self.thresholds), 3), np.int64)  # tp, fp, fn
        self.ois_f1, self.biou, self.hd95 = [], [], []

    def update(self, prob, gt, valid=None):
        prob, gt = np.asarray(prob, np.float32), np.asarray(gt, bool)
        if valid is not None:
            valid = np.asarray(valid, bool)
            prob = np.where(valid, prob, 0.0)
            gt = gt & valid

        best = 0.0
        for i, t in enumerate(self.thresholds):
            tp, fp, fn = match_counts(prob >= t, gt, self.tolerance)
            self.counts[i] += (tp, fp, fn)
            best = max(best, prf_from_counts(tp, fp, fn)[2])
        self.ois_f1.append(best)

        pred05 = prob >= 0.5
        self.biou.append(boundary_iou(pred05, gt, self.boundary_dilation))
        self.hd95.append(hausdorff_percentile(pred05, gt))

    def compute(self):
        f1s = [prf_from_counts(*self.counts[i])[2] for i in range(len(self.thresholds))]
        best = int(np.argmax(f1s))
        p, r, f = prf_from_counts(*self.counts[best])
        return {
            "ODS_F1": float(f),
            "ODS_threshold": float(self.thresholds[best]),
            "ODS_precision": float(p),
            "ODS_recall": float(r),
            "OIS_F1": _nanmean(self.ois_f1),
            "BoundaryIoU": _nanmean(self.biou),
            "HD95_px": _nanmean(self.hd95),
        }


def pixels_to_angle_arcsec(px: float, focal_len_mm: float, pixel_size_mm: float) -> float:
    """Convert a pixel localisation error to an angular error in arcseconds.

    This is the bridge from a vision metric to a navigation one.  For the
    project's camera (35 mm lens, 4.96 um pixels -> f = 7056 px) one native
    pixel is about 29 arcsec.
    """
    f_px = focal_len_mm / pixel_size_mm
    return float(np.degrees(np.arctan2(px, f_px)) * 3600.0)


def limb_error_to_range_error_km(px: float, limb_radius_px: float, range_km: float) -> float:
    """First-order range error from a limb-radius error of ``px`` pixels.

    ``d_range ~ range * (d_r / r)``.  At 75,000 km with a 602 px limb radius,
    one native pixel is roughly 125 km; the same error at a 256x256 downsample
    (75 px radius) is roughly 1000 km, which is why output resolution is a
    navigation parameter and not a training convenience.
    """
    if limb_radius_px <= 0:
        return float("nan")
    return float(range_km * px / limb_radius_px)
