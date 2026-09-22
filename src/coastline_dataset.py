"""Dataset for Earth land/water/cloud segmentation and boundary detection.

Replaces the original 3-class dataset.  The substantive changes:

* **Aspect ratio is preserved.**  The renders are 2048x1536 (4:3).  The old
  code resized to a square 256x256, which compresses x by 8 and y by 6 and
  turns the Earth disc into an ellipse with a 1.33 axis ratio.  That silently
  breaks the camera model every downstream optical-navigation routine in
  ``functions.py`` assumes (``ellipse_fit``, ``christian_robinson``,
  ``pos_estimation``).  The default target is now 240x320, which is the same
  4:3 and divisible by 16, and a mismatched aspect ratio raises instead of
  quietly distorting.

* **Labels come from ``labels.py``** - 5 classes and 3 boundary channels with
  per-channel validity, instead of a Sobel filter over a shaded mask.

* **Augmentation.**  None existed.  A spacecraft sees arbitrary roll, varying
  exposure and sensor noise, none of which appeared in training.  This is
  domain randomisation in the sense of Tobin et al., IROS 2017
  (arXiv:1703.06907).

* **Deterministic, stable train/val split** by filename hash, so adding images
  does not reshuffle which frames are held out.

* **File pairing is checked**, rather than assuming a naming convention and
  crashing (or silently mispairing) when it does not hold.
"""

from __future__ import annotations

import hashlib
import os
import re
import warnings
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import labels as L

#: Native render size is 4:3 (functions.py, case_type 3000: 2048x1536).
DEFAULT_IMG_SIZE = (240, 320)  # (H, W)

#: Per-channel mean/std over the synthetic set.  Recompute with
#: ``python src/compute_class_weights.py --stats`` when the data changes.
DEFAULT_MEAN = (0.0975, 0.1021, 0.1188)
DEFAULT_STD = (0.1560, 0.1552, 0.1725)


@dataclass
class AugmentConfig:
    """Augmentation switches.  All off reproduces plain resize + normalise."""

    enabled: bool = True
    #: Spacecraft roll about the boresight is unconstrained, so full 360 deg.
    rotate: bool = True
    #: Mirroring is not physical for Earth, but it is harmless for *detecting*
    #: coast/limb/terminator (as opposed to identifying which landmass) and it
    #: is a cheap regulariser.  Turn off if you move to landmark identification.
    flip: bool = True
    brightness: float = 0.25
    contrast: float = 0.25
    gamma: float = 0.25
    gaussian_noise_std: float = 0.02
    shot_noise: bool = True
    blur_sigma_max: float = 1.2
    p_photometric: float = 0.8
    p_noise: float = 0.5
    p_blur: float = 0.3


@dataclass
class DatasetConfig:
    image_dir: str = "dataset/images"
    mask_dir: str = "dataset/masks"
    cloud_mask_dir: str = "dataset/cloud masks"
    img_size: tuple = DEFAULT_IMG_SIZE
    night_luma: float = L.DEFAULT_NIGHT_LUMA
    mean: tuple = DEFAULT_MEAN
    std: tuple = DEFAULT_STD
    val_fraction: float = 0.2
    split_seed: int = 1337
    allow_anisotropic: bool = False
    allow_legacy_masks: bool = False
    augment: AugmentConfig = field(default_factory=AugmentConfig)


_IMG_RE = re.compile(r"^earth_img_(?P<idx>.+)\.png$")


def _stable_seed(name: str, seed: int) -> int:
    """Reproducible 32-bit seed for one sample in one epoch."""
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _stable_bucket(name: str, seed: int) -> float:
    """Deterministic [0, 1) value for a filename - stable across runs, machines
    and Python versions (unlike ``hash()``)."""
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64


class CoastlineDataset(Dataset):
    """Paired RGB / land-sea / cloud renders -> segmentation + boundary targets.

    Each item is a dict:

    ``image``      float32 ``(3, H, W)``, normalised
    ``seg``        int64 ``(H, W)`` with values from ``labels``
    ``edges``      float32 ``(3, H, W)`` boundary targets
    ``edge_valid`` float32 ``(3, H, W)`` per-channel supervision mask
    ``name``       source filename, for error reporting and qualitative figures
    """

    def __init__(self, config: DatasetConfig | None = None, split: str = "all", **kwargs):
        self.cfg = config or DatasetConfig(**kwargs)
        if split not in ("all", "train", "val"):
            raise ValueError(f"split must be all/train/val, got {split!r}")
        self.split = split

        cfg = self.cfg
        for d in (cfg.image_dir, cfg.mask_dir, cfg.cloud_mask_dir):
            if not os.path.isdir(d):
                raise FileNotFoundError(f"missing dataset directory: {d}")

        self.samples = self._index()
        if not self.samples:
            raise RuntimeError(
                f"no usable image/mask/cloud triples found under {cfg.image_dir!r} "
                f"for split={split!r}"
            )
        self._checked_sentinel = False

    # ---------------------------------------------------------------- indexing
    def _index(self):
        cfg = self.cfg
        names = sorted(
            f for f in os.listdir(cfg.image_dir)
            if f.endswith(".png") and "MASK" not in f
        )

        samples, missing = [], []
        for name in names:
            m = _IMG_RE.match(name)
            if not m:
                missing.append((name, "filename does not match earth_img_<idx>.png"))
                continue
            idx = m.group("idx")
            mask_p = os.path.join(cfg.mask_dir, f"earth_img_MASK{idx}.png")
            cloud_p = os.path.join(cfg.cloud_mask_dir, f"earth_img_CLOUDMASK{idx}.png")
            if not os.path.exists(mask_p):
                missing.append((name, f"no land/sea mask at {mask_p}"))
                continue
            if not os.path.exists(cloud_p):
                missing.append((name, f"no cloud mask at {cloud_p}"))
                continue
            samples.append((name, os.path.join(cfg.image_dir, name), mask_p, cloud_p))

        if missing:
            preview = "\n  ".join(f"{n}: {why}" for n, why in missing[:5])
            warnings.warn(
                f"skipping {len(missing)} image(s) with no usable mask pair:\n  {preview}"
                + ("\n  ..." if len(missing) > 5 else ""),
                stacklevel=2,
            )

        if self.split == "all":
            return samples
        want_val = self.split == "val"
        return [
            s for s in samples
            if (_stable_bucket(s[0], self.cfg.split_seed) < self.cfg.val_fraction) == want_val
        ]

    def __len__(self):
        return len(self.samples)

    # ------------------------------------------------------------------- io
    def _read_rgb(self, path):
        arr = cv2.imread(path, cv2.IMREAD_COLOR)
        if arr is None:
            raise RuntimeError(f"could not read image: {path}")
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    def _check_aspect(self, h, w):
        th, tw = self.cfg.img_size
        src, dst = w / h, tw / th
        if abs(src - dst) / src > 0.01 and not self.cfg.allow_anisotropic:
            raise ValueError(
                f"anisotropic resize refused: source is {w}x{h} (aspect {src:.4f}) but "
                f"img_size is {tw}x{th} (aspect {dst:.4f}).\n"
                f"Squashing the frame turns the Earth disc into an ellipse and breaks "
                f"the camera model used by ellipse_fit/christian_robinson in "
                f"functions.py. Pick a target with the same aspect ratio (e.g. "
                f"{DEFAULT_IMG_SIZE[1]}x{DEFAULT_IMG_SIZE[0]}), or pass "
                f"allow_anisotropic=True if you really mean it."
            )

    # ----------------------------------------------------------- augmentation
    def _augment_geometry(self, image, seg, rng):
        aug = self.cfg.augment
        if aug.flip:
            if rng.random() < 0.5:
                image, seg = image[:, ::-1], seg[:, ::-1]
            if rng.random() < 0.5:
                image, seg = image[::-1], seg[::-1]
            image, seg = np.ascontiguousarray(image), np.ascontiguousarray(seg)

        if aug.rotate:
            angle = rng.uniform(0.0, 360.0)
            h, w = seg.shape
            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
            # Empty frame really is more empty space, so a black / SPACE fill
            # is physically consistent rather than a padding hack.
            image = cv2.warpAffine(
                image, M, (w, h), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=(0.0, 0.0, 0.0),
            )
            seg = cv2.warpAffine(
                seg, M, (w, h), flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=int(L.SPACE),
            )
        return image, seg

    def _augment_photometric(self, image, rng):
        aug = self.cfg.augment
        if rng.random() < aug.p_photometric:
            if aug.brightness:
                image = image + rng.uniform(-aug.brightness, aug.brightness)
            if aug.contrast:
                f = 1.0 + rng.uniform(-aug.contrast, aug.contrast)
                image = (image - 0.5) * f + 0.5
            image = np.clip(image, 0.0, 1.0)
            if aug.gamma:
                g = float(np.exp(rng.uniform(-aug.gamma, aug.gamma)))
                image = np.power(image, g)

        if rng.random() < aug.p_blur and aug.blur_sigma_max > 0:
            sigma = rng.uniform(0.3, aug.blur_sigma_max)
            image = cv2.GaussianBlur(image, (0, 0), sigma)

        if rng.random() < aug.p_noise:
            if aug.shot_noise:
                # Poisson-like: variance grows with signal, as on a real sensor.
                scale = 400.0
                image = rng.poisson(np.clip(image, 0, 1) * scale) / scale
            if aug.gaussian_noise_std:
                image = image + rng.normal(0.0, aug.gaussian_noise_std, image.shape)

        return np.clip(image, 0.0, 1.0).astype(np.float32)

    # ---------------------------------------------------------------- getitem
    def __getitem__(self, idx):
        name, img_p, mask_p, cloud_p = self.samples[idx]
        image = self._read_rgb(img_p)
        mask = self._read_rgb(mask_p)
        cloud = self._read_rgb(cloud_p)

        if image.shape[:2] != mask.shape[:2] or image.shape[:2] != cloud.shape[:2]:
            raise RuntimeError(
                f"{name}: size mismatch - image {image.shape[:2]}, "
                f"mask {mask.shape[:2]}, cloud {cloud.shape[:2]}"
            )

        if not self._checked_sentinel:
            if not (L.has_space_sentinel(mask) or self.cfg.allow_legacy_masks):
                raise RuntimeError(
                    f"{name}: this mask has no space sentinel, so it was rendered "
                    f"before the label fix in functions.py. Those masks put empty "
                    f"space and ocean at the same value and shade the mask by "
                    f"sunlight, which labels the night side as water and the "
                    f"terminator as a coastline. Re-render the dataset, or pass "
                    f"allow_legacy_masks=True to train on known-bad labels."
                )
            self._checked_sentinel = True

        h, w = image.shape[:2]
        self._check_aspect(h, w)

        # Compose at native resolution: the night test reads the crisp render,
        # and the mask boundaries have not yet been resampled.
        seg = L.compose_seg_label(image, mask, cloud, night_luma=self.cfg.night_luma)

        th, tw = self.cfg.img_size
        image = cv2.resize(image, (tw, th), interpolation=cv2.INTER_AREA)
        seg = cv2.resize(seg, (tw, th), interpolation=cv2.INTER_NEAREST)

        if self.cfg.augment.enabled and self.split != "val":
            # hashlib, not hash(): Python randomises string hashing per
            # process unless PYTHONHASHSEED is set, which would make augmented
            # runs unreproducible.
            rng = np.random.default_rng(_stable_seed(name, torch.initial_seed()))
            image, seg = self._augment_geometry(image, seg, rng)
            image = self._augment_photometric(image, rng)

        # Derive boundaries from the final label map so targets and labels can
        # never disagree, and so every boundary is exactly one pixel wide.
        edges, edge_valid = L.make_edge_targets(seg)

        image = (image - np.asarray(self.cfg.mean, np.float32)) / np.asarray(
            self.cfg.std, np.float32
        )

        return {
            "image": torch.from_numpy(image.transpose(2, 0, 1).copy()),
            "seg": torch.from_numpy(seg.astype(np.int64)),
            "edges": torch.from_numpy(edges),
            "edge_valid": torch.from_numpy(edge_valid),
            "name": name,
        }
