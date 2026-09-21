"""
Checks the ground-truth labels themselves, independent of any trained model.

Answers three questions we've been guessing at from prediction outputs alone:
  1. Does the land/water mask convention actually match what the code assumes
     (white=land, black=water)? Checked objectively via average RGB color per
     mask class - water should look blue, land should not.
  2. How much of the dataset is actually cloud, water, land? (real numbers,
     not the back-solved guess from class_weights)
  3. How much true water/land gets hidden (overwritten to class "cloud") by
     the cloud mask?

Also saves a handful of RGB / land-water-mask / cloud-mask side-by-side
panels so you can eyeball a few images directly.

No model, no training - reads dataset/images, dataset/masks, dataset/cloud
masks directly. Safe/cheap to run any time.
"""
import os

import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt

IMAGE_DIR = "dataset/images"
MASK_DIR = "dataset/masks"
CLOUD_MASK_DIR = "dataset/cloud masks"
IMG_SIZE = (256, 256)
N_SAMPLE_PANELS = 8  # how many side-by-side images to save for manual inspection
OUT_DIR = "label_audit_samples"
BACKGROUND_BRIGHTNESS_THRESHOLD = 0.02  # must match coastline_dataset.py's IGNORE_INDEX threshold

image_transform = T.Compose([T.Resize(IMG_SIZE), T.ToTensor()])
mask_transform = T.Compose([
    T.Resize(IMG_SIZE, interpolation=InterpolationMode.NEAREST),
    T.Grayscale(num_output_channels=1),
    T.ToTensor(),
])

image_files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.endswith(".png") and "MASK" not in f
])
print(f"Found {len(image_files)} images in {IMAGE_DIR}\n")

# running totals across the whole dataset
total_pixels = 0
count_water_final = 0  # seg_label == 0 (after cloud overwrite)
count_land_final = 0   # seg_label == 1
count_cloud_final = 0  # seg_label == 2

count_water_raw = 0  # land_water_bin == 0, before cloud overwrite
count_land_raw = 0   # land_water_bin == 1

water_hidden_by_cloud = 0  # true water (raw) that cloud mask overwrote
land_hidden_by_cloud = 0   # true land (raw) that cloud mask overwrote

count_background = 0   # final "water" pixels that are actually near-black empty space
count_true_water = 0   # final "water" pixels that are real, visible ocean

# color accumulators: sum of RGB values + pixel count, per class, restricted to
# NON-cloud pixels for water/land so cloud doesn't contaminate the color signal
rgb_sum_water = np.zeros(3)
rgb_count_water = 0
rgb_sum_land = np.zeros(3)
rgb_count_land = 0
rgb_sum_cloud = np.zeros(3)
rgb_count_cloud = 0

os.makedirs(OUT_DIR, exist_ok=True)

for idx, img_name in enumerate(image_files):
    number = img_name.replace("earth_img_", "").replace(".png", "")
    mask_path = os.path.join(MASK_DIR, f"earth_img_MASK{number}.png")
    cloud_path = os.path.join(CLOUD_MASK_DIR, f"earth_img_CLOUDMASK{number}.png")
    img_path = os.path.join(IMAGE_DIR, img_name)

    if not (os.path.exists(mask_path) and os.path.exists(cloud_path)):
        print(f"  skipping {img_name}: missing mask or cloud mask file")
        continue

    image = Image.open(img_path).convert("RGB")
    land_water = Image.open(mask_path).convert("L")
    cloud = Image.open(cloud_path).convert("L")

    image_t = image_transform(image)  # (3, H, W), 0-1
    land_water_bin = (mask_transform(land_water) > 0.5).squeeze(0).numpy()  # (H, W) bool
    cloud_bin = (mask_transform(cloud) > 0.5).squeeze(0).numpy()

    image_np = image_t.permute(1, 2, 0).numpy()  # (H, W, 3)

    n_pix = land_water_bin.size
    total_pixels += n_pix

    raw_water_mask = ~land_water_bin
    raw_land_mask = land_water_bin

    count_water_raw += raw_water_mask.sum()
    count_land_raw += raw_land_mask.sum()

    water_hidden_by_cloud += (raw_water_mask & cloud_bin).sum()
    land_hidden_by_cloud += (raw_land_mask & cloud_bin).sum()

    final_water_mask = raw_water_mask & ~cloud_bin
    final_land_mask = raw_land_mask & ~cloud_bin
    final_cloud_mask = cloud_bin

    count_water_final += final_water_mask.sum()
    count_land_final += final_land_mask.sum()
    count_cloud_final += final_cloud_mask.sum()

    brightness = image_np.mean(axis=2)  # (H, W)
    background_mask = final_water_mask & (brightness < BACKGROUND_BRIGHTNESS_THRESHOLD)
    true_water_mask = final_water_mask & ~background_mask
    count_background += background_mask.sum()
    count_true_water += true_water_mask.sum()

    rgb_sum_water += image_np[final_water_mask].sum(axis=0)
    rgb_count_water += final_water_mask.sum()
    rgb_sum_land += image_np[final_land_mask].sum(axis=0)
    rgb_count_land += final_land_mask.sum()
    rgb_sum_cloud += image_np[final_cloud_mask].sum(axis=0)
    rgb_count_cloud += final_cloud_mask.sum()

    if idx < N_SAMPLE_PANELS:
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        axes[0].imshow(image_np); axes[0].set_title("RGB"); axes[0].axis("off")
        axes[1].imshow(land_water_bin, cmap="gray"); axes[1].set_title("Land/water mask (white=?)"); axes[1].axis("off")
        axes[2].imshow(cloud_bin, cmap="gray"); axes[2].set_title("Cloud mask (white=cloud)"); axes[2].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"panel_{img_name}"), dpi=100)
        plt.close(fig)

print(f"Processed {len(image_files)} images, {total_pixels} total pixels\n")

print("=== Real pixel-class frequency (after cloud overwrite, i.e. what the model actually trains on) ===")
print(f"  water: {count_water_final/total_pixels*100:.1f}%")
print(f"  land:  {count_land_final/total_pixels*100:.1f}%")
print(f"  cloud: {count_cloud_final/total_pixels*100:.1f}%")

print("\n=== How much true water/land gets hidden by the cloud mask ===")
print(f"  {water_hidden_by_cloud/max(count_water_raw,1)*100:.1f}% of raw water pixels overwritten to 'cloud'")
print(f"  {land_hidden_by_cloud/max(count_land_raw,1)*100:.1f}% of raw land pixels overwritten to 'cloud'")

pct_background_of_water = count_background / max(count_water_final, 1) * 100
print(f"\n=== How much of 'water' is actually empty background space, not ocean ===")
print(f"  {pct_background_of_water:.1f}% of pixels labeled 'water' are near-black (brightness < {BACKGROUND_BRIGHTNESS_THRESHOLD}) - likely background, not real ocean")
print(f"  (this is what coastline_dataset.py's IGNORE_INDEX now excludes from training)")

# corrected frequency: same denominator, but true_water instead of water_final, background dropped entirely
total_non_background = total_pixels - count_background
print("\n=== Corrected pixel-class frequency (background excluded, matches what the model now actually trains on) ===")
pct_true_water = count_true_water / total_non_background * 100
pct_land = count_land_final / total_non_background * 100
pct_cloud = count_cloud_final / total_non_background * 100
print(f"  water: {pct_true_water:.1f}%")
print(f"  land:  {pct_land:.1f}%")
print(f"  cloud: {pct_cloud:.1f}%")

# recommended sqrt-inverse-frequency weights from the CORRECTED numbers, normalized to sum=3
freqs = {"water": pct_true_water / 100, "land": pct_land / 100, "cloud": pct_cloud / 100}
inv_sqrt = {k: 1 / (v ** 0.5) for k, v in freqs.items() if v > 0}
total_inv_sqrt = sum(inv_sqrt.values())
recommended = {k: v / total_inv_sqrt * 3 for k, v in inv_sqrt.items()}
print(f"\n  Recommended class_weights (paste into losses.py):")
print(f"  torch.tensor([{recommended['water']:.3f}, {recommended['land']:.3f}, {recommended['cloud']:.3f}])  # water, land, cloud")

print("\n=== Average RGB per class (non-cloud pixels only for water/land) ===")
if rgb_count_water > 0:
    r, g, b = rgb_sum_water / rgb_count_water
    print(f"  water-labeled pixels: R={r:.3f} G={g:.3f} B={b:.3f}  {'(blue-dominant, looks like water - good)' if b > r and b > g else '(NOT blue-dominant - mask convention may be WRONG)'}")
if rgb_count_land > 0:
    r, g, b = rgb_sum_land / rgb_count_land
    print(f"  land-labeled pixels:  R={r:.3f} G={g:.3f} B={b:.3f}  {'(NOT blue-dominant - looks right for land)' if not (b > r and b > g) else '(blue-dominant - mask convention may be WRONG, this looks like water)'}")
if rgb_count_cloud > 0:
    r, g, b = rgb_sum_cloud / rgb_count_cloud
    brightness = (r + g + b) / 3
    print(f"  cloud-labeled pixels: R={r:.3f} G={g:.3f} B={b:.3f}  (brightness={brightness:.3f}, {'looks bright/white - good' if brightness > 0.6 else 'NOT very bright - check cloud mask quality'})")

print(f"\nSaved {min(N_SAMPLE_PANELS, len(image_files))} side-by-side panels to {OUT_DIR}/ for manual inspection")
