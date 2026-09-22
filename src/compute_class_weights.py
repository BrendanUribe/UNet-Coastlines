"""Derive class weights and normalisation statistics from the actual dataset.

The original code carried ``class_weights = [0.075, 2.109, 0.816]`` and
``pos_weight = 31.2`` as bare literals with no comment and no script that
produced them.  Nothing in the repository could regenerate them, so they went
stale silently the moment the dataset changed - and they were computed over a
class 0 that conflated ocean with empty space, so they were wrong regardless.

Run this after regenerating the dataset and paste the output into the config:

    python src/compute_class_weights.py --config configs/default.yaml
    python src/compute_class_weights.py --config configs/default.yaml --stats

Median-frequency balancing follows Eigen & Fergus, ICCV 2015 (and SegNet,
Badrinarayanan et al., IEEE TPAMI 39(12), 2017): ``w_c = median(f) / f_c``.
It is less explosive than raw inverse frequency when one class - here empty
space - dominates the frame.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import labels as L
from coastline_dataset import AugmentConfig, CoastlineDataset, DatasetConfig


def build_dataset(args):
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        d = cfg["data"]
        ds_cfg = DatasetConfig(
            image_dir=d["image_dir"], mask_dir=d["mask_dir"],
            cloud_mask_dir=d["cloud_mask_dir"], img_size=tuple(d["img_size"]),
            night_luma=d["night_luma"], val_fraction=d["val_fraction"],
            split_seed=d["split_seed"],
            # Statistics must describe the raw data, so no augmentation, and
            # normalisation is disabled by using identity mean/std.
            mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0),
            augment=AugmentConfig(enabled=False),
        )
    else:
        ds_cfg = DatasetConfig(
            image_dir=args.image_dir, mask_dir=args.mask_dir,
            cloud_mask_dir=args.cloud_mask_dir,
            mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0),
            augment=AugmentConfig(enabled=False),
        )
    return CoastlineDataset(ds_cfg, args.split)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--image-dir", default="dataset/images")
    ap.add_argument("--mask-dir", default="dataset/masks")
    ap.add_argument("--cloud-mask-dir", default="dataset/cloud masks")
    ap.add_argument("--split", default="train", choices=["train", "val", "all"])
    ap.add_argument("--scheme", default="median",
                    choices=["median", "inverse", "sqrt_inverse"])
    ap.add_argument("--stats", action="store_true",
                    help="also compute per-channel mean/std for normalisation")
    ap.add_argument("--max-images", type=int, default=0)
    args = ap.parse_args()

    ds = build_dataset(args)
    n = len(ds) if not args.max_images else min(len(ds), args.max_images)
    print(f"scanning {n} images from the {args.split} split ...")

    counts = np.zeros(L.NUM_SEG_CLASSES, np.int64)
    edge_pos = np.zeros(L.NUM_EDGE_CHANNELS, np.int64)
    edge_valid = np.zeros(L.NUM_EDGE_CHANNELS, np.int64)
    ch_sum = np.zeros(3, np.float64)
    ch_sq = np.zeros(3, np.float64)
    n_px = 0

    for i in range(n):
        s = ds[i]
        seg = s["seg"].numpy()
        counts += np.bincount(seg.ravel(), minlength=L.NUM_SEG_CLASSES)

        e, v = s["edges"].numpy(), s["edge_valid"].numpy()
        edge_pos += ((e > 0.5) & (v > 0.5)).sum(axis=(1, 2)).astype(np.int64)
        edge_valid += (v > 0.5).sum(axis=(1, 2)).astype(np.int64)

        if args.stats:
            img = s["image"].numpy()           # identity normalisation -> raw [0,1]
            ch_sum += img.sum(axis=(1, 2))
            ch_sq += (img ** 2).sum(axis=(1, 2))
            n_px += img.shape[1] * img.shape[2]

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n}")

    freq = counts / counts.sum()
    present = freq > 0

    if args.scheme == "median":
        weights = np.where(present, np.median(freq[present]) / np.maximum(freq, 1e-12), 0.0)
    elif args.scheme == "inverse":
        weights = np.where(present, 1.0 / np.maximum(freq, 1e-12), 0.0)
    else:
        weights = np.where(present, 1.0 / np.sqrt(np.maximum(freq, 1e-12)), 0.0)
    weights = weights / weights[present].mean()   # mean 1 -> loss scale unchanged

    print("\nClass frequencies and weights"
          f" ({args.scheme}-frequency balancing)")
    print(f"  {'class':8s} {'pixels':>12s} {'fraction':>10s} {'weight':>9s}")
    for c in range(L.NUM_SEG_CLASSES):
        flag = "" if present[c] else "   <- ABSENT from this split"
        print(f"  {L.SEG_CLASS_NAMES[c]:8s} {counts[c]:12d} {freq[c]:10.5f} "
              f"{weights[c]:9.4f}{flag}")
    if not present.all():
        print("  WARNING: a class never occurs. Its weight is 0 and the model "
              "cannot learn it -\n           regenerate the dataset with more "
              "varied geometry before training.")

    print("\nBoundary channel statistics (valid pixels only)")
    for c in range(L.NUM_EDGE_CHANNELS):
        frac = edge_pos[c] / max(edge_valid[c], 1)
        pw = (1 - frac) / frac if frac > 0 else float("inf")
        print(f"  {L.EDGE_CHANNEL_NAMES[c]:11s} positive fraction {frac:.6f}  "
              f"-> implied pos_weight {pw:8.1f}")
    print("  (losses.py estimates this per batch, so it does not need to be "
          "configured;\n   it is printed here to make the imbalance visible.)")

    print("\nPaste into configs/*.yaml under loss:")
    print("  class_weights: [" + ", ".join(f"{w:.4f}" for w in weights) + "]")

    if args.stats:
        mean = ch_sum / n_px
        std = np.sqrt(np.maximum(ch_sq / n_px - mean ** 2, 1e-12))
        print("\nPaste into configs/*.yaml under data:")
        print("  mean: [" + ", ".join(f"{m:.4f}" for m in mean) + "]")
        print("  std:  [" + ", ".join(f"{s:.4f}" for s in std) + "]")


if __name__ == "__main__":
    main()
