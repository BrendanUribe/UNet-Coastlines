"""Quantitative evaluation over a held-out split.

This is the script that produces reportable numbers.  It prints, per class,
IoU and Dice; per boundary channel, ODS/OIS F1, Boundary IoU and the 95th
percentile Hausdorff distance; and converts that last one into the angular and
range errors it implies for optical navigation, so a vision metric can be read
as a navigation budget.

    python src/evaluate.py --checkpoint runs/default/best.pth --split val
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import labels as L
from coastline_dataset import AugmentConfig, CoastlineDataset, DatasetConfig
from metrics import EdgeMetrics, SegMetrics, limb_error_to_range_error_km, pixels_to_angle_arcsec
from UNet import UNet


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="val", choices=["val", "train", "all"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tolerance", type=float, default=2.0,
                    help="boundary matching tolerance in pixels")
    ap.add_argument("--json-out", default=None)
    # Camera, for the navigation conversion. Defaults are case_type 3000.
    ap.add_argument("--focal-len-mm", type=float, default=35.0)
    ap.add_argument("--pixel-size-mm", type=float, default=4.96e-3)
    ap.add_argument("--native-width", type=int, default=2048)
    ap.add_argument("--range-km", type=float, default=75000.0)
    ap.add_argument("--earth-radius-km", type=float, default=6378.1)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    d = cfg.get("data", {})

    ds_cfg = DatasetConfig(
        image_dir=d.get("image_dir", "dataset/images"),
        mask_dir=d.get("mask_dir", "dataset/masks"),
        cloud_mask_dir=d.get("cloud_mask_dir", "dataset/cloud masks"),
        img_size=tuple(d.get("img_size", (240, 320))),
        night_luma=d.get("night_luma", L.DEFAULT_NIGHT_LUMA),
        mean=tuple(d.get("mean", (0.0975, 0.1021, 0.1188))),
        std=tuple(d.get("std", (0.1560, 0.1552, 0.1725))),
        val_fraction=d.get("val_fraction", 0.2),
        split_seed=d.get("split_seed", 1337),
        augment=AugmentConfig(enabled=False),   # never augment at evaluation
    )
    ds = CoastlineDataset(ds_cfg, args.split)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    model = UNet(in_channels=3, num_classes=L.NUM_SEG_CLASSES,
                 num_edge_channels=L.NUM_EDGE_CHANNELS, **cfg.get("model", {})).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    seg_m = SegMetrics(L.NUM_SEG_CLASSES, L.SEG_CLASS_NAMES)
    edge_m = {c: EdgeMetrics(tolerance=args.tolerance) for c in range(L.NUM_EDGE_CHANNELS)}

    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device)
            seg_out, edge_out, _ = model(image)
            seg_m.update(seg_out.argmax(1).cpu().numpy(), batch["seg"].numpy())
            prob = torch.sigmoid(edge_out).cpu().numpy()
            e, v = batch["edges"].numpy(), batch["edge_valid"].numpy()
            for b in range(image.size(0)):
                for c in range(L.NUM_EDGE_CHANNELS):
                    edge_m[c].update(prob[b, c], e[b, c] > 0.5, v[b, c] > 0.5)

    seg_res = seg_m.compute()
    print(f"\n=== {args.split} split: {len(ds)} images, checkpoint epoch "
          f"{ckpt.get('epoch','?')} ===\n")
    print("Segmentation")
    print(f"  pixel accuracy {seg_res['pixel_acc']:.4f}")
    print(f"  mIoU           {seg_res['mIoU']:.4f}")
    print(f"  mDice          {seg_res['mDice']:.4f}")
    for c in range(L.NUM_SEG_CLASSES):
        print(f"    IoU {L.SEG_CLASS_NAMES[c]:6s} {seg_res[f'IoU/{L.SEG_CLASS_NAMES[c]}']:.4f}")

    # The evaluation runs at the training resolution; a pixel there is several
    # native pixels, which is what actually sets the navigation error.
    th, tw = ds_cfg.img_size
    downsample = args.native_width / tw
    f_px_native = args.focal_len_mm / args.pixel_size_mm
    limb_radius_native = f_px_native * np.tan(
        np.arcsin(args.earth_radius_km / args.range_km)
    )
    limb_radius_eval = limb_radius_native / downsample

    print("\nBoundaries")
    edge_res = {}
    for c in range(L.NUM_EDGE_CHANNELS):
        r = edge_m[c].compute()
        edge_res[L.EDGE_CHANNEL_NAMES[c]] = r
        print(f"  {L.EDGE_CHANNEL_NAMES[c]}")
        print(f"    ODS F1        {r['ODS_F1']:.4f}  (threshold {r['ODS_threshold']:.2f}, "
              f"P {r['ODS_precision']:.3f} / R {r['ODS_recall']:.3f})")
        print(f"    OIS F1        {r['OIS_F1']:.4f}")
        print(f"    Boundary IoU  {r['BoundaryIoU']:.4f}")
        print(f"    HD95          {r['HD95_px']:.2f} px (eval grid)")

    limb = edge_res[L.EDGE_CHANNEL_NAMES[L.EDGE_LIMB]]
    hd = limb["HD95_px"]
    nav = {}
    if np.isfinite(hd):
        hd_native = hd * downsample
        def rng_err(px_eval=None, px_native=None):
            if px_native is not None:
                return limb_error_to_range_error_km(
                    px_native, limb_radius_native, args.range_km)
            return limb_error_to_range_error_km(px_eval, limb_radius_eval, args.range_km)

        nav = {
            "hd95_eval_px": hd,
            "downsample_factor": downsample,
            "hd95_native_px": hd_native,
            "angular_error_arcsec": pixels_to_angle_arcsec(
                hd_native, args.focal_len_mm, args.pixel_size_mm),
            "limb_radius_native_px": float(limb_radius_native),
            "limb_radius_eval_px": float(limb_radius_eval),
            "range_error_km_achieved": rng_err(px_eval=hd),
            # Floors: the best any method could do at each localisation
            # precision. These are what the resolution choice actually costs;
            # the achieved error above is what the network currently adds.
            "range_error_km_floor_1px_eval": rng_err(px_eval=1.0),
            "range_error_km_floor_1px_native": rng_err(px_native=1.0),
            "range_error_km_floor_subpixel": rng_err(px_native=0.1),
        }
        print(f"\nNavigation implication (limb channel, {args.range_km:,.0f} km, "
              f"f={f_px_native:.0f} px)")
        print(f"  limb radius: {limb_radius_eval:.1f} px on the {tw}x{th} eval grid, "
              f"{limb_radius_native:.1f} px native ({downsample:.1f}x downsample)")
        print(f"  achieved HD95 {hd:.2f} eval px = {hd_native:.1f} native px "
              f"= {nav['angular_error_arcsec']:,.0f} arcsec")
        print(f"    -> range error {nav['range_error_km_achieved']:,.0f} km")
        print("  floors set by localisation precision alone:")
        print(f"    1 px on the eval grid            "
              f"{nav['range_error_km_floor_1px_eval']:,.0f} km")
        print(f"    1 px at native resolution        "
              f"{nav['range_error_km_floor_1px_native']:,.0f} km")
        print(f"    0.1 px, sub-pixel limb fitting   "
              f"{nav['range_error_km_floor_subpixel']:,.0f} km")
        print("  The first two floors differ by the downsample factor: an argmax mask"
              "\n  on the eval grid cannot beat the first line no matter how well the"
              "\n  network is trained. Sub-pixel limb localisation is what reaches the"
              "\n  third (Christian, JSR 54(3), 2017; Christian, IEEE Access 9, 2021).")

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump({"split": args.split, "n_images": len(ds),
                       "checkpoint": args.checkpoint, "epoch": ckpt.get("epoch"),
                       "segmentation": seg_res, "boundaries": edge_res,
                       "navigation": nav}, f, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
