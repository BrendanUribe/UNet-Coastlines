"""Qualitative prediction on a single frame.

The original hardcoded a checkpoint filename that still contained its
``2025-XX-XX`` placeholder, so the script could not run as committed; it also
computed a cloud-suppressed coastline and then never displayed it.

This version takes arguments, restores the checkpoint's own configuration so
the model is rebuilt exactly as trained, and renders the four panels.  For
numbers over a held-out set use ``evaluate.py`` - a single eyeballed frame is
not a result.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import labels as L
from UNet import UNet

#: Display colours (RGB) for the segmentation classes.
CLASS_COLOURS = np.array(
    [
        [0, 0, 0],        # space
        [20, 70, 160],    # water
        [90, 150, 60],    # land
        [235, 235, 240],  # cloud
        [45, 40, 70],     # night
    ],
    dtype=np.uint8,
)

EDGE_COLOURS = {
    L.EDGE_COASTLINE: (255, 60, 60),
    L.EDGE_LIMB: (60, 255, 120),
    L.EDGE_TERMINATOR: (255, 210, 60),
}


def load_model(ckpt_path, device):
    """Rebuild the model from the checkpoint's stored config."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})

    model = UNet(
        in_channels=3,
        num_classes=L.NUM_SEG_CLASSES,
        num_edge_channels=L.NUM_EDGE_CHANNELS,
        **model_cfg,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    img_size = tuple(data_cfg.get("img_size", (240, 320)))
    mean = np.asarray(data_cfg.get("mean", (0.0975, 0.1021, 0.1188)), np.float32)
    std = np.asarray(data_cfg.get("std", (0.1560, 0.1552, 0.1725)), np.float32)
    return model, img_size, mean, std, ckpt.get("epoch")


def preprocess(path, img_size, mean, std):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    th, tw = img_size
    src, dst = rgb.shape[1] / rgb.shape[0], tw / th
    if abs(src - dst) / src > 0.01:
        print(
            f"WARNING: source aspect {src:.3f} != target {dst:.3f}; the Earth disc "
            f"will be distorted and any limb fit from this frame will be biased."
        )
    resized = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA)
    norm = (resized - mean) / std
    return resized, torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0)


def overlay(rgb01, edge_masks, alpha=1.0):
    out = (np.clip(rgb01, 0, 1) * 255).astype(np.uint8).copy()
    for ch, mask in edge_masks.items():
        out[mask] = (
            (1 - alpha) * out[mask] + alpha * np.array(EDGE_COLOURS[ch], np.float32)
        ).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True, help="path to best.pth / last.pth")
    ap.add_argument("--image", required=True, help="RGB render to run on")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--save", default=None, help="write the figure here instead of showing it")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--sweep", action="store_true",
                    help="render the coastline channel at a range of thresholds; "
                         "use it to find the operating point before retraining")
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, img_size, mean, std, epoch = load_model(args.checkpoint, device)
    print(f"loaded {args.checkpoint} (epoch {epoch}) on {device}, input {img_size}")

    rgb, tensor = preprocess(args.image, img_size, mean, std)
    tensor = tensor.to(device)

    t0 = time.time()
    with torch.inference_mode():
        seg_out, edge_out, _ = model(tensor)
        seg = seg_out.argmax(1)[0].cpu().numpy()
        edge_prob = torch.sigmoid(edge_out)[0].cpu().numpy()
    infer_s = time.time() - t0

    raw = {c: edge_prob[c] >= args.threshold for c in range(L.NUM_EDGE_CHANNELS)}

    # A coastline cannot be observed through cloud or past the terminator, so
    # suppress predictions there rather than drawing them.
    unobservable = np.isin(seg, [L.CLOUD, L.NIGHT, L.SPACE])
    clean = dict(raw)
    clean[L.EDGE_COASTLINE] = raw[L.EDGE_COASTLINE] & ~unobservable

    print(f"inference: {infer_s*1000:.1f} ms on {device} "
          f"({'GPU' if device.type=='cuda' else 'CPU'}; state the hardware when quoting this)")
    total = seg.size
    for c in range(L.NUM_SEG_CLASSES):
        print(f"  {L.SEG_CLASS_NAMES[c]:6s} {(seg==c).sum()/total:6.3f}")
    for c in range(L.NUM_EDGE_CHANNELS):
        print(f"  {L.EDGE_CHANNEL_NAMES[c]:11s} {int(clean[c].sum()):6d} px "
              f"(raw {int(raw[c].sum())})")

    import matplotlib
    if args.no_show or args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.sweep:
        thresholds = [0.5, 0.7, 0.85, 0.95, 0.99]
        fig, axes = plt.subplots(1, len(thresholds) + 1, figsize=(4 * (len(thresholds) + 1), 4.5))
        axes[0].imshow(rgb); axes[0].set_title("RGB"); axes[0].axis("off")
        for ax, t in zip(axes[1:], thresholds):
            m = (edge_prob[L.EDGE_COASTLINE] >= t) & ~unobservable
            over = overlay(rgb, {L.EDGE_COASTLINE: m})
            ax.imshow(over)
            ax.set_title(f"coastline @ {t:.2f}\n{int(m.sum())} px")
            ax.axis("off")
        fig.suptitle("Coastline channel vs threshold - pick where lines stop being blobs")
        fig.tight_layout()
        if args.save:
            os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
            fig.savefig(args.save, dpi=140, bbox_inches="tight")
            print(f"sweep written to {args.save}")
        elif not args.no_show:
            plt.show()
        return

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(rgb); axes[0].set_title("RGB")
    axes[1].imshow(CLASS_COLOURS[seg]); axes[1].set_title("Segmentation")
    bands = np.zeros((*seg.shape, 3), np.uint8)
    for c, m in clean.items():
        bands[m] = EDGE_COLOURS[c]
    axes[2].imshow(bands)
    axes[2].set_title("Boundaries\nred=coast  green=limb  yellow=terminator")
    axes[3].imshow(overlay(rgb, clean)); axes[3].set_title("Boundaries over RGB")
    for a in axes:
        a.axis("off")
    fig.tight_layout()

    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        fig.savefig(args.save, dpi=140, bbox_inches="tight")
        print(f"figure written to {args.save}")
    elif not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
