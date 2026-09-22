"""Audit a rendered dataset before training on it.

Checks each image/mask/cloud-mask triple for the problems that are cheap to
find now and expensive to find after a training run:

* masks rendered before the label fix (no space sentinel) - these put empty
  space and ocean at the same value and shade the mask by sunlight
* a mix of pre-fix and post-fix renders in the same folder, which is what you
  get when the generator is re-run without clearing the output directory
* missing or mismatched pairs
* frames with no illuminated surface at all

Read-only unless --quarantine is passed.

    python src/check_dataset.py
    python src/check_dataset.py --quarantine dataset/_legacy
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import labels as L


def read_rgb(path):
    arr = cv2.imread(path, cv2.IMREAD_COLOR)
    if arr is None:
        return None
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--image-dir", default="dataset/images")
    ap.add_argument("--mask-dir", default="dataset/masks")
    ap.add_argument("--cloud-mask-dir", default="dataset/cloud masks")
    ap.add_argument("--quarantine", default=None,
                    help="move pre-fix / broken triples into this directory")
    ap.add_argument("--list-all", action="store_true")
    args = ap.parse_args()

    names = sorted(f for f in os.listdir(args.image_dir)
                   if f.endswith(".png") and "MASK" not in f)
    if not names:
        print(f"no images found in {args.image_dir}")
        return 1

    ok, legacy, broken = [], [], []
    night_seen = False
    for name in names:
        idx = name[len("earth_img_"):-len(".png")]
        mp = os.path.join(args.mask_dir, f"earth_img_MASK{idx}.png")
        cp = os.path.join(args.cloud_mask_dir, f"earth_img_CLOUDMASK{idx}.png")
        ip = os.path.join(args.image_dir, name)

        if not os.path.exists(mp):
            broken.append((name, "no land/sea mask")); continue
        if not os.path.exists(cp):
            broken.append((name, "no cloud mask")); continue

        img, mask, cloud = read_rgb(ip), read_rgb(mp), read_rgb(cp)
        if img is None or mask is None or cloud is None:
            broken.append((name, "unreadable png")); continue
        if img.shape[:2] != mask.shape[:2] or img.shape[:2] != cloud.shape[:2]:
            broken.append((name, f"size mismatch {img.shape[:2]}/{mask.shape[:2]}/{cloud.shape[:2]}"))
            continue

        if not L.has_space_sentinel(mask):
            legacy.append((name, "mask has no space sentinel (pre-fix render)"))
            continue
        if not L.has_space_sentinel(cloud):
            legacy.append((name, "cloud mask has no space sentinel (pre-fix render)"))
            continue

        seg = L.compose_seg_label(img, mask, cloud)
        present = set(np.unique(seg).tolist())
        if L.NIGHT in present:
            night_seen = True
        if not ({L.WATER, L.LAND, L.CLOUD} & present):
            broken.append((name, "no illuminated surface in frame")); continue
        ok.append(name)

    total = len(names)
    print(f"\n{total} images in {args.image_dir}")
    print(f"  usable (post-fix): {len(ok)}")
    print(f"  pre-fix renders:   {len(legacy)}")
    print(f"  broken/unpaired:   {len(broken)}")

    if legacy:
        print("\nPRE-FIX RENDERS - these carry the space-as-water and")
        print("terminator-as-coastline defects and must not be trained on:")
        for n, why in (legacy if args.list_all else legacy[:10]):
            print(f"  {n}: {why}")
        if not args.list_all and len(legacy) > 10:
            print(f"  ... and {len(legacy)-10} more (--list-all to see them)")

    if broken:
        print("\nBROKEN / UNPAIRED:")
        for n, why in (broken if args.list_all else broken[:10]):
            print(f"  {n}: {why}")
        if not args.list_all and len(broken) > 10:
            print(f"  ... and {len(broken)-10} more")

    if ok and not night_seen:
        print("\nWARNING: no frame contains an unilluminated surface. Every render is")
        print("full-phase, so the night/terminator handling is untested and the")
        print("'night' class will be absent. Generate some higher phase angles.")

    if legacy and args.quarantine:
        for sub in ("images", "masks", "cloud masks"):
            os.makedirs(os.path.join(args.quarantine, sub), exist_ok=True)
        moved = 0
        for name, _ in legacy:
            idx = name[len("earth_img_"):-len(".png")]
            trio = [(args.image_dir, name, "images"),
                    (args.mask_dir, f"earth_img_MASK{idx}.png", "masks"),
                    (args.cloud_mask_dir, f"earth_img_CLOUDMASK{idx}.png", "cloud masks")]
            for src_dir, fn, sub in trio:
                src = os.path.join(src_dir, fn)
                if os.path.exists(src):
                    shutil.move(src, os.path.join(args.quarantine, sub, fn))
            moved += 1
        print(f"\nmoved {moved} pre-fix triples to {args.quarantine}")
        print(f"{len(ok)} usable images remain")

    if legacy and not args.quarantine:
        print("\nTo set them aside (nothing is deleted):")
        print("  python src/check_dataset.py --quarantine dataset/_legacy")
        print("Then re-render those indices, or train on what is left.")

    return 0 if (ok and not legacy and not broken) else 1


if __name__ == "__main__":
    sys.exit(main())
