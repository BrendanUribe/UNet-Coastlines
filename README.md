# UNet-Coastlines

Land / water / cloud segmentation and coastline, limb and terminator detection
on synthetic Earth imagery, aimed at autonomous optical navigation for a
spacecraft.

Images are rendered in POV-Ray with SPICE ephemerides, so every frame comes
with exact ground truth for the spacecraft state — which means a detection can
be scored not just as a segmentation but as a **navigation measurement**.

---

## Quick start

```bash
pip install -r requirements.txt

# 1. generate renders (needs POV-Ray + SPICE kernels + texture maps)
python src/main_image_gen_SC_coast_local.py

# 2. derive class weights and normalisation stats from your data
python src/compute_class_weights.py --config configs/default.yaml --stats
#    paste the two blocks it prints into configs/default.yaml

# 3. train
python src/train_unet.py --config configs/default.yaml

# 4. score a held-out split
python src/evaluate.py --checkpoint runs/default/best.pth --split val

# 5. look at one frame
python src/predict_unet.py --checkpoint runs/default/best.pth \
                           --image dataset/images/earth_img_2.png
```

Tests (no POV-Ray, SPICE or GPU required):

```bash
python -m pytest tests/ -v
```

---

## Labels

Five segmentation classes and three boundary channels, defined in
[`src/labels.py`](src/labels.py), which is the single source of truth for
label semantics.

| id | class | meaning |
|----|-------|---------|
| 0 | `space` | empty sky |
| 1 | `water` | illuminated ocean |
| 2 | `land`  | illuminated land |
| 3 | `cloud` | illuminated cloud |
| 4 | `night` | Earth surface past the terminator |

| id | boundary | why it is separate |
|----|----------|--------------------|
| 0 | `coastline` | landmark matching |
| 1 | `limb` | the illuminated horizon — the conic that is fitted for a position fix |
| 2 | `terminator` | must be **excluded** from a limb fit, so it has to be detected rather than ignored |

Each boundary channel carries a validity mask. A coast under cloud or past the
terminator is not observable, so it is neither rewarded nor punished; likewise
the unlit limb, which is real geometry but carries no signal.

### Why the label scheme changed

The previous pipeline used three classes derived by running a Sobel filter over
a *shaded* render of the land-sea mask. That produced three label defects, all
of which the network faithfully learned:

1. **Space was labelled as water.** POV-Ray's default background is black,
   which is the same value the mask used for ocean.
2. **The night side was labelled as water, and night clouds vanished.** The
   masks were rendered with `ambient 0 / diffuse 1`, i.e. lit by the Sun. With
   `brilliance 0` the falloff is a step, so the terminator became a hard edge
   in the mask — and therefore a **coastline label**. That is the long arc that
   appeared across the disc in predictions.
3. **The cloud sphere was clipped to the Sun-facing hemisphere**, which is not
   the camera-facing hemisphere at high phase angle (the generator admits up to
   135°).

The renderer now emits pure geometry (`ambient 1 / diffuse 0`) on a sentinel
background, and illumination comes back in as its own class, recovered from the
RGB image. The relevant edits are marked `LABEL-CORRECTNESS FIX` in
[`src/functions.py`](src/functions.py).

**Masks rendered before this fix are refused at load time** rather than
silently mislabelled. Re-render, or pass `allow_legacy_masks=True` if you
deliberately want to reproduce the old behaviour.

---

## Resolution and the camera model

Renders are 2048×1536 (4:3). The default training size is **240×320**, which is
the same aspect ratio and divisible by 16.

The previous default was 256×256, which compresses x by 8 and y by 6 and turns
the Earth disc into an ellipse with a 1.33 axis ratio. That silently breaks the
camera model that every optical-navigation routine in `functions.py` assumes
(`ellipse_fit`, `christian_robinson`, `pos_estimation`). The dataset now
**raises** on an aspect-changing resize instead of performing it.

Resolution is a navigation parameter, not a training convenience. With the
project camera (35 mm, 4.96 µm pixels → *f* = 7056 px) at 75,000 km, the limb
radius is 602 px native and 94 px at 240×320:

| limb localisation | range error at 75,000 km |
|---|---|
| 1 px on the 240×320 grid | ~800 km |
| 1 px native | ~125 km |
| 0.1 px, sub-pixel fitting | ~12 km |

`src/evaluate.py` prints this table for your actual checkpoint.

---

## Architecture

`src/UNet.py` — U-Net (Ronneberger et al., MICCAI 2015) with a segmentation
head and a deeply-supervised boundary head. Changes from the 2015 formulation:

- **GroupNorm** instead of no normalisation. The original had none, which is
  why the learning rate was pinned at 5e-5 to stay stable; the default is now
  1e-3. GroupNorm rather than BatchNorm because batch sizes here are small
  (4–8), exactly where batch statistics degrade (Wu & He, ECCV 2018).
- **Depth 4** instead of 3 — a whole-disc view needs global context. Depth 4 is
  the maximum at 240×320 (240 = 16 × 15).
- **Bilinear upsample + conv** instead of `ConvTranspose2d`, which produces
  periodic checkerboard artifacts (Odena et al., Distill 2016) — unaffordable
  when the output is one-pixel-wide lines.
- **Deep supervision** on the boundary head, per HED (Xie & Tu, ICCV 2015).

| `base_width` | depth | params |
|---|---|---|
| 64 | 4 | 28.9 M |
| 32 | 4 | 7.2 M |
| 16 | 4 | 1.8 M |

Use 32 or 16 for an embedded target and report parameters, MACs and inference
time on representative hardware.

## Loss

`src/losses.py` — cross-entropy + soft Dice for segmentation, class-balanced
BCE for boundaries, with the two tasks weighted by **learned homoscedastic
uncertainty** (Kendall, Gal & Cipolla, CVPR 2018) rather than summed 1:1. The
boundary positive weight is estimated per batch instead of hardcoded.

The reported **total loss can go negative** once the task losses drop below the
regularisation term. That is expected for this objective; judge progress from
the per-task `seg` and `edge` figures.

## Metrics

`src/metrics.py`. Plain IoU is reported but is the wrong headline number for a
one-pixel-wide structure. Quote instead:

- **Boundary IoU** (Cheng et al., CVPR 2021)
- **ODS / OIS F1** with tolerance matching (BSDS protocol, Arbeláez et al.,
  TPAMI 2011)
- **95th-percentile Hausdorff distance** — worst-case localisation error in
  pixels, which converts directly into an angular and range error

---

## Repository layout

```
configs/default.yaml   training configuration; a run is fully described by
                       this file plus the git commit
src/labels.py          label semantics — classes, boundaries, validity
src/coastline_dataset.py  loading, aspect-safe resize, augmentation, splits
src/UNet.py            model
src/losses.py          CE + Dice + balanced BCE, learned task weighting
src/metrics.py         segmentation and boundary metrics, nav conversions
src/train_unet.py      training entry point
src/evaluate.py        held-out metrics and navigation implications
src/predict_unet.py    single-frame qualitative figure
src/compute_class_weights.py  regenerate class weights and normalisation stats
tests/                 regression tests for the label and metric logic
src/functions.py       POV-Ray + SPICE + OPNAV library (see Provenance)
src/main_image_gen_*.py, src/preprocess_earth_local.py   image generation
```

## Known gaps

Work that is understood but not yet done, in priority order:

1. **The navigation back-end is not connected.** `functions.py` already
   contains `ellipse_fit` (4458), `christian_robinson` (4873) and
   `pos_estimation` (4573) — an implementation of Christian & Robinson,
   JGCD 39(12), 2016 — and nothing in the learning pipeline calls them. Closing
   the loop (network → limb → conic fit → position vs. SPICE truth, reported in
   km) is what turns this from a segmentation result into an aerospace one.
2. **Boundaries are thresholded, not localised to sub-pixel.** `(prob > 0.5)`
   discards the precision optical navigation depends on.
3. **No uncertainty estimate.** A navigation filter needs a measurement
   covariance; networks are overconfident by default (Guo et al., ICML 2017).
   MC dropout (Gal & Ghahramani, ICML 2016) or deep ensembles
   (Lakshminarayanan et al., NeurIPS 2017).
4. **The committed generator config is a debug one** — `num_theta=1`,
   `num_phi=2`, `dist_all=[75000]`. A single range means the model never sees
   the Earth at a different apparent size. Restore the full grid and sample
   viewpoints equal-area on the sphere rather than on a lat/lon grid, which
   over-samples the poles.
5. **No quantised or embedded benchmark.** See Jacob et al., CVPR 2018.

## Provenance

`src/functions.py` and the `main_image_gen_*` / `preprocess_earth_local`
scripts originate with **Tim Kilduff (UC San Diego, 2023–24)**, with
Dr. Pablo Machuca (MIT/SDSU) and Dr. Aaron J. Rosengren (UCSD). See the header
of `src/preprocess_earth_local.py`. That code is treated as a vendored
dependency: apart from the marked label-correctness fixes it is deliberately
left unrefactored. Settle licensing and attribution before publishing.

## References

**Architecture**
- Ronneberger, Fischer & Brox. *U-Net.* MICCAI 2015. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
- Wu & He. *Group Normalization.* ECCV 2018. [arXiv:1803.08494](https://arxiv.org/abs/1803.08494)
- Odena, Dumoulin & Olah. *Deconvolution and Checkerboard Artifacts.* Distill, 2016.
- Isensee et al. *nnU-Net.* Nature Methods 18, 203–211, 2021.

**Joint segmentation and edges**
- Heidler, Mou, Baumhoer, Dietz & Zhu. *HED-UNet: Combined Segmentation and Edge Detection for Monitoring the Antarctic Coastline.* IEEE TGRS 60, 2022. [arXiv:2103.01849](https://arxiv.org/abs/2103.01849) — the closest published analogue to this architecture.
- Xie & Tu. *Holistically-Nested Edge Detection.* ICCV 2015. [arXiv:1504.06375](https://arxiv.org/abs/1504.06375)
- Takikawa et al. *Gated-SCNN.* ICCV 2019. [arXiv:1907.05740](https://arxiv.org/abs/1907.05740)
- Kendall, Gal & Cipolla. *Multi-Task Learning Using Uncertainty to Weigh Losses.* CVPR 2018. [arXiv:1705.07115](https://arxiv.org/abs/1705.07115)
- Milletari et al. *V-Net.* 3DV 2016. [arXiv:1606.04797](https://arxiv.org/abs/1606.04797)
- Kervadec et al. *Boundary loss.* Medical Image Analysis 67, 2021. [arXiv:1812.07032](https://arxiv.org/abs/1812.07032)

**Metrics**
- Cheng, Girshick, Dollár, Kirillov & He. *Boundary IoU.* CVPR 2021. [arXiv:2103.16562](https://arxiv.org/abs/2103.16562)
- Arbeláez, Maire, Fowlkes & Malik. *Contour Detection and Hierarchical Image Segmentation.* IEEE TPAMI 33(5), 2011.

**Optical navigation**
- Christian. *A Tutorial on Horizon-Based Optical Navigation and Attitude Determination With Space Imaging Systems.* IEEE Access 9, 19819–19853, 2021.
- Christian. *Accurate Planetary Limb Localization for Image-Based Spacecraft Navigation.* JSR 54(3), 2017.
- Christian & Robinson. *Noniterative Horizon-Based Optical Navigation by Cholesky Factorization.* JGCD 39(12), 2016.
- Mortari, D'Souza & Zanetti. *Image Processing of Illuminated Ellipsoid.* JSR 53(3), 2016.
- Owen. *Methods of Optical Navigation.* AAS 11-215, 2011.

**Learned landmark detection for navigation**
- Silburt et al. *Lunar crater identification via deep learning.* Icarus 317, 27–38, 2019. [arXiv:1803.02192](https://arxiv.org/abs/1803.02192)
- Downes, Steiner & How. *Deep Learning Crater Detection for Lunar Terrain Relative Navigation.* AIAA SciTech 2020.
- Pugliatti & Topputo. *Navigation about irregular bodies through segmentation maps.* AAS/AIAA 2021.
- Campbell, Furfaro, Linares & Gaylor. *A Deep Learning Approach for Optical Autonomous Planetary Relative Terrain Navigation.* AAS 2017.

**Sim-to-real and deployment**
- Tobin et al. *Domain Randomization.* IROS 2017. [arXiv:1703.06907](https://arxiv.org/abs/1703.06907)
- Giuffrida et al. *The Φ-Sat-1 Mission: The First On-Board Deep Neural Network Demonstrator for Satellite Earth Observation.* IEEE TGRS 60, 2022.
- Furano et al. *Towards the Use of Artificial Intelligence on the Edge in Space Systems.* IEEE A&E Systems Magazine 35(12), 2020.
- Jacob et al. *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.* CVPR 2018. [arXiv:1712.05877](https://arxiv.org/abs/1712.05877)
- Guo et al. *On Calibration of Modern Neural Networks.* ICML 2017. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)
- Gal & Ghahramani. *Dropout as a Bayesian Approximation.* ICML 2016. [arXiv:1506.02142](https://arxiv.org/abs/1506.02142)
- Lakshminarayanan et al. *Deep Ensembles.* NeurIPS 2017. [arXiv:1612.01474](https://arxiv.org/abs/1612.01474)
