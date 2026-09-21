# Passdown: `UNet-Land/Cloud/Water-working`

**Purpose:** brief a new Claude Code session (or a human) on exactly where this branch stands, what was broken, what was fixed, why, and what's still open. Read this before touching `src/`.

**Project context:** two-headed U-Net (segmentation: land/water/cloud + edge: coastline) for detecting Earth's coastline from optical images, for Brendan's M.S. Aerospace Engineering (GNC) thesis work — eventual goal is feeding detected coastline landmarks into an optical-navigation pipeline (satellite/cislunar). This branch is purely the perception model; navigation/estimation work has not started.

**User's working style — read this first:**
- Wants deep, mechanistic explanations ("teach me like I don't understand it"), not hand-waved summaries. Exact variable semantics, exact code location, numeric walkthroughs.
- **Never push without explicit confirmation.** A previous unwanted push was reverted (`git revert --no-edit`, not reset/force-push) after the user said "I didn't want to push that part yet." Hold changes locally and ask before pushing, even though the branch task framework says to push when done — this user has explicitly overridden that once already.
- Wants problems root-caused empirically (via scripts/audits reading real data), never diagnosed by guessing from how a prediction image looks.
- Wants every multi-file code change actually executed (forward/backward pass, dummy tensors, gradient/NaN checks) before being trusted — this caught 4 real bugs this cycle (see below). Don't skip this step.

---

## 1. Two problems solved this cycle

### Problem 1 — model predicted "cloud" instead of "water" on real ocean

**Root cause (found via `src/audit_dataset_labels.py`, not guessing):** the land/water mask gives identical labels to open ocean *and* the black empty space surrounding the Earth disc in these renders. Measured average RGB of "water"-labeled pixels: (0.038, 0.041, 0.048) — essentially black, not blue. **88.3% of everything labeled "water" was actually background space.** This is why an earlier attempt to fix this via loss reweighting alone didn't work — it reinforced "black space = water," not "bright ocean = water."

**Fix:**
- `src/coastline_dataset.py`: added `IGNORE_INDEX = -100` and `BACKGROUND_BRIGHTNESS_THRESHOLD = 0.02`. In `__getitem__`, pixels that are near-black (`brightness < 0.02`) AND not land AND not cloud get `seg_label[is_background] = IGNORE_INDEX` — excluded from training entirely, not counted as any class.
- `src/losses.py`: `CrossEntropyLoss(weight=class_weights, ignore_index=-100)` — those pixels contribute zero gradient.
- Recomputed class weights from the corrected, background-excluded frequencies: water 43.7% / land 17.4% / cloud 38.9% (down from a wildly imbalanced raw split that was mostly background noise).
- **Extended background exclusion to inference too** (`src/predict_unet.py`), since the model gets zero training signal on background and its raw prediction there is arbitrary: `background_pixels = (image_plot.mean(axis=2) < 0.02)` computed directly from the input image, then used to force those pixels to neutral gray in the segmentation display and to zero out `edge_prob` there before thinning.

### Problem 2 — predicted coastline was a thick blob, not a thin line

**Root cause:** two structural properties of the original loss (not a training bug): `pos_weight=31.2` set the effective decision threshold at only ~3% confidence (`1/(1+pos_weight)`), and plain per-pixel BCE has no concept of "thinness" — it grades pixels independently with no penalty for an oversized region.

**Fixes (all four, `src/UNet.py` + `src/losses.py` + `src/predict_unet.py`):**
1. Softened `pos_weight` 31.2 → 5.6 (sqrt of original — same softening technique as the class weights below).
2. Added Dice loss alongside BCE (`dice_loss()` in `losses.py`) — penalizes area/overlap mismatch directly, not just per-pixel correctness.
3. **Multi-scale ("HED-style") supervision** — `UNet.py` now has three edge heads (`edge_head_d3`, `edge_head_d2`, `edge_head_d1`) at the coarse/medium/fine decoder stages, each upsampled to full resolution via `F.interpolate` and combined through a learned 1×1 fusion conv (`edge_fuse`). `forward()` returns `seg_out, [edge_d3_up, edge_d2_up, edge_d1, edge_fused]` — **a list of 4 tensors, not one tensor.** Anything that consumes the model's edge output must account for this (see bugs below).
4. Replaced `skeletonize` (topological thinning) with **non-max suppression** (`non_max_suppress_thin()` in `predict_unet.py`) — same second-stage algorithm as classic Canny edge detection (Canny, 1986, IEEE PAMI). Computes Sobel gradient direction on the *predicted probability map*, keeps only pixels that are local maxima along that direction. Runs on the continuous probability map; thresholding (`> 0.5`) happens *after* thinning, not before.

---

## 2. Current file states (verified working, verified by real execution)

- **`src/UNet.py`** — multi-scale edge heads as described above. Requires `import torch.nn.functional as F` (was missing initially, caused `NameError`).
- **`src/losses.py`** — `combined_loss(seg_out, edge_outputs, seg_label, edge_label, cloud_label)`. `edge_outputs` is the 4-item list; loops over all 4 and averages `edge_loss_single()` (BCE + Dice each) so every decoder stage actually trains on edges, not just the fused output. `class_weights = torch.tensor([0.824, 1.304, 0.873])` — see §4 for exact derivation.
- **`src/predict_unet.py`** — loads model, runs inference, applies cloud-suppression (`edge_prob[seg_pred==2] = 0`) and background-suppression, runs NMS thinning, thresholds, plots a 4-panel figure (RGB / segmentation / thin coastline / overlay). Unpacks `seg_out, edge_outputs = model(...)`, indexes `edge_outputs[-1]` for the fused tensor.
- **`src/train_unet.py`** — unpacks `seg_out, edge_outputs = model(images)`, passes list into `combined_loss`. `epochs = 50`. No resume-from-checkpoint logic on this branch.
- **`src/coastline_dataset.py`** — `IGNORE_INDEX`/background exclusion as above; also generates edge ground truth via Sobel on the binary land/water mask (`make_edge_mask`).
- **`src/audit_dataset_labels.py`** — diagnostic-only script, no model needed. Reports real class frequency, background contamination %, per-class average RGB, recommended weights. Re-run this any time the dataset changes — don't assume the 43.7/17.4/38.9 split still holds if images are added.

**Architecture changed this cycle (new multi-scale edge heads) → no existing `.pth` checkpoint is compatible. A full training run from scratch is required; none has been completed yet on this exact code.**

---

## 3. Bugs already found and fixed this cycle (don't reintroduce these)

- Missing `import torch.nn.functional as F` in `UNet.py` → `NameError` on `F.interpolate`.
- `H. W = edge_prob.shape` (period instead of comma) in `predict_unet.py` — valid Python syntax (parses as attribute assignment), only fails at runtime. Lesson: "it compiles" ≠ "it's correct," especially for typo'd tuple unpacking.
- `torch.sigmoid(edge_outputs)` called on the raw list instead of `edge_outputs[-1]` — recurred more than once. `edge_outputs` is always a 4-item list now; anything downstream needs `[-1]` for the fused output or to loop over all 4.
- `losses.py`'s `combined_loss` initially still did `edge_out.squeeze(1)` on the list directly (a leftover from the single-tensor design) — would crash with `AttributeError: 'list' object has no attribute 'squeeze'` on first real batch. Fixed by extracting `edge_loss_single()` and averaging over the list.
- `dice_loss` originally referenced an undefined `pred_prob` variable and would have double-applied sigmoid to an already-sigmoided input (silently wrong numbers, no crash). Fixed.
- Background brightness threshold accidentally set to `0.2` (10x too loose — would misclassify real dark ocean/land as background) instead of `0.02`. Must match between `coastline_dataset.py` (training) and `predict_unet.py` (inference).

---

## 4. Class weight math (exact derivation, for reference)

Corrected frequencies (background excluded): water 0.437, land 0.174, cloud 0.389.

1. Raw inverse frequency (`1/freq`): water 2.288, land 5.747, cloud 2.571 — too aggressive (land/water ratio ~2.5x here; was far worse, ~20x+, before background exclusion, because raw water frequency was inflated by background pixels).
2. Softened via square root (`1/sqrt(freq)`): water 1.5127, land 2.3976, cloud 1.6034 — same softening principle as SegNet's median-frequency balancing / Cui et al.'s effective-number weighting.
3. Normalized to mean 1 (divide by 1.8379): **water 0.8235, land 1.3045, cloud 0.8724** → matches `class_weights` in `losses.py` exactly.

---

## 5. Known open issue — NOT yet fixed

**Red coastline pixels sometimes appear over cloud regions despite `edge_prob[cloud_pixels] = 0`.** Diagnosed but not fixed: the suppression only works where `seg_pred` correctly calls a pixel cloud (class 2). Cloud *edges/boundaries* are exactly where segmentation is most likely to flicker between classes (ambiguous, semi-transparent cloud), and those same boundary pixels are exactly where the edge head tends to be most confident (real intensity gradients that resemble a coastline). Net effect: gaps in the cloud mask at cloud boundaries let spurious edge confidence leak through.

Two fix options discussed, not yet implemented:
- **Cheap/inference-only:** dilate `cloud_pixels` a few pixels (`cv2.dilate`) before suppressing, to cover segmentation flicker at cloud boundaries. Quick to test on the current (untrained) architecture once a checkpoint exists.
- **Real fix, needs retraining:** couple the two heads — feed `seg_out` (or its softmax) into the edge branch, or add an explicit loss term penalizing edge confidence wherever ground-truth cloud label is 1. Bigger change, not started.

---

## 6. Reference citations — verification status (for the professor write-up)

Verified via live web search this cycle:
- ✅ Canny (1986), *A Computational Approach to Edge Detection*, IEEE PAMI — correct, NMS is literally step 3 of Canny's algorithm.
- ✅ Badrinarayanan et al., *SegNet*, arXiv:1511.00561 — correct.
- ✅ Cui et al., *Class-Balanced Loss Based on Effective Number of Samples*, CVPR 2019, arXiv:1901.05555 — correct.
- ⚠️ Hosseini & Baghshah, arXiv:2412.06045 (*Dilated Balanced Cross Entropy Loss*) — real paper, but it does NOT argue "raw inverse-frequency weighting overcorrects" (it argues the opposite framing: plain balanced CE degrades performance, proposes dilated-mask weighting instead). Should be reframed or dropped from the write-up.
- ⚠️ Heidler et al., *HED-UNet*, arXiv:2103.01849 — correct paper and correctly the architecture this project is modeled on, but they merge multi-scale outputs with a **learned hierarchical attention mechanism**, not a plain 1×1 fusion conv like `edge_fuse`. Write-up should say "inspired by HED-UNet's multi-scale supervision, using a simpler fixed fusion layer," not imply the fusion mechanism itself was replicated.

Reference doc: `/tmp/deckwork/Coastline_UNet_Professor_Writeup.md` (not in this repo — scratch/deliverable file) still has both of these framed incorrectly as of last edit; fix pending if the user wants it corrected before sharing.

---

## 7. Status and next steps (in priority order)

**Nothing above has been validated on a real full training run yet.** Everything is confirmed *mechanically correct* (forward pass, multi-scale loss computation, backward pass, all params receiving gradients, no NaNs, on synthetic data) but not confirmed to actually produce better results on real data.

Next steps, in order:
1. **Establish a held-out validation split** if one doesn't already exist — check `train_unet.py`/dataset loading before assuming one is there.
2. **Run a full training cycle from scratch** (architecture changed, no compatible checkpoint).
3. **Quantitative validation**, not eyeballing: per-class IoU/Dice for segmentation, a confusion matrix, and buffered edge precision/recall (dilate GT coastline ±2-3px, standard BSDS-style tolerance) for the coastline. Currently there is no evaluation script — `predict_unet.py` only visualizes one image at a time.
4. **Log per-component loss curves** (`seg_loss`, and ideally each of the 4 edge-scale losses separately, not just the summed total) — `train_unet.py` currently only tracks the combined total.
5. **Add a ground-truth comparison panel** to `predict_unet.py` (show `edge_label` next to `edge_thin` for the same image) — right now there's no way to see prediction vs. actual side by side.
6. **Test across image diversity** (heavy cloud, small islands, different zoom/distance scales), not just one repeatedly-viewed test image.
7. Revisit the still-open finding: land is disproportionately cloud-occluded in ground truth (43% vs 6% for water) — may starve the land class the same way the original water problem worked; check land recall specifically on cloud-adjacent land pixels after the next training run.
8. Address §5 above (cloud-boundary coastline leak) once a trained checkpoint exists to test against.
9. Dataset scale: currently ~200 images, pipeline targets ~1000 — hold off scaling until the corrected pipeline is confirmed working on the smaller set.

---

*Generated as a session handoff. Not committed/pushed — ask the user before adding this to git history.*
