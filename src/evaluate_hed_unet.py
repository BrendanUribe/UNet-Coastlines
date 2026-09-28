# scores the trained HED-UNet on the VALIDATION set (images the model never trained on)
# prints averages over all images and saves one overlay plot per image + a summary table
# make the validation set with:  python src/make_label_masks.py validation

import os # file finding/paths
import csv # summary table
import numpy as np # math
import torch
import matplotlib
matplotlib.use("Agg") # save plots to files, no windows
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF # image to tensor
from scipy.spatial import cKDTree # nearest-point distances for coastline error
from hed_unet import HEDUNet # HED-UNet model
from coastline_dataset import letterbox # same no-stretch resize + padding as training
from coastline_lines import predicted_line, true_boundary_points, MRAD_PER_PIXEL # thin lines + true coastline
from train_hed_unet import CLASS_NAMES, NUM_CLASSES, IMG_SIZE, MODEL_PATH, prepare_labels # same settings as training

VAL_IMAGE_DIR = "dataset/validation_images"
VAL_LABEL_DIR = "dataset/validation_labels"
RESULTS_DIR = "dataset/validation_results" # overlay plots + summary.csv saved here

print("STARTED")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HEDUNet(in_channels=3, out_channels=NUM_CLASSES + 1).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device)) # name set in train_hed_unet.py
model.eval()

image_files = sorted(
    f for f in os.listdir(VAL_IMAGE_DIR)
    if f.startswith("earth_img_") and f.endswith(".png") and f[len("earth_img_"):-len(".png")].replace("_", "").isdigit()
)
print(f"Evaluating {len(image_files)} validation images with {MODEL_PATH} (IMG_SIZE {IMG_SIZE})\n")

# totals over all images
intersections = np.zeros(NUM_CLASSES)
unions = np.zeros(NUM_CLASSES)
correct_pixels = total_pixels = 0
all_errors = [] # sub-pixel line point -> nearest true coastline, original pixels
all_found = [] # per true coastline point: line point within 1 model pixel?
rows = []

for img_name in image_files:
    number = img_name[len("earth_img_"):-len(".png")]
    image_full = Image.open(os.path.join(VAL_IMAGE_DIR, img_name)).convert("RGB")
    label_full = Image.open(os.path.join(VAL_LABEL_DIR, f"earth_img_LABEL{number}.png"))

    image_small, (scale, pad_x, pad_y) = letterbox(image_full, IMG_SIZE, Image.BILINEAR)
    label_small, _ = letterbox(label_full, IMG_SIZE, Image.NEAREST)
    truth = prepare_labels(np.array(label_small))
    truth_full = prepare_labels(np.array(label_full))

    with torch.inference_mode():
        combined, _ = model(TF.to_tensor(image_small).unsqueeze(0).to(device))
        class_map = combined[0, :NUM_CLASSES].argmax(dim=0).cpu().numpy()
        edge_prob = torch.sigmoid(combined[0, NUM_CLASSES]).cpu().numpy()
    edge_band = (edge_prob > 0.5).astype(np.uint8)

    # class scores on the real image area only (not the letterbox padding)
    valid = np.zeros_like(truth, dtype=bool)
    new_w, new_h = round(image_full.width / scale), round(image_full.height / scale)
    valid[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = True
    for c in range(NUM_CLASSES):
        pred_c, truth_c = (class_map == c) & valid, (truth == c) & valid
        intersections[c] += np.logical_and(pred_c, truth_c).sum()
        unions[c] += np.logical_or(pred_c, truth_c).sum()
    image_correct = ((class_map == truth) & valid).sum()
    correct_pixels += image_correct
    total_pixels += valid.sum()

    # coastline: thin sub-pixel line vs exact true boundary, in original pixels
    segments, line_full, _ = predicted_line(edge_band, edge_prob, scale, pad_x, pad_y)
    true_boundary = true_boundary_points(truth_full)
    row = dict(image=img_name, pixel_accuracy=image_correct / valid.sum(), true_coast_points=len(true_boundary),
               line_points=len(line_full), median_error_px="", median_error_mrad="", coast_found_pct="")
    if len(true_boundary) and len(line_full):
        errors = cKDTree(true_boundary).query(line_full)[0]
        found = cKDTree(line_full).query(true_boundary)[0] <= scale
        all_errors.append(errors)
        all_found.append(found)
        row.update(median_error_px=np.median(errors), median_error_mrad=np.median(errors) * MRAD_PER_PIXEL,
                   coast_found_pct=100 * found.mean())
        print(f"{img_name}: pixel acc {row['pixel_accuracy']:.3f}, coastline error median {np.median(errors):.2f} px, "
              f"found {100 * found.mean():.0f}%")
    else:
        print(f"{img_name}: pixel acc {row['pixel_accuracy']:.3f}, no visible coastline (true or predicted)")
    rows.append(row)

    # overlay plot saved per image: true coastline, predicted band, predicted thin line
    ys, xs = np.nonzero(edge_band)
    plt.figure(figsize=(12, 9))
    plt.imshow(np.array(image_full))
    plt.scatter((xs - pad_x + 0.5) * scale - 0.5, (ys - pad_y + 0.5) * scale - 0.5, s=4, c="red", marker="s",
                linewidths=0, alpha=0.5, label="predicted band (model pixels)")
    if len(true_boundary):
        plt.scatter(true_boundary[:, 0], true_boundary[:, 1], s=0.3, c="lime", linewidths=0, label="true coastline")
    for i, seg in enumerate(segments):
        plt.plot(seg[:, 0], seg[:, 1], c="cyan", linewidth=1, label="predicted line (thinned, sub-pixel)" if i == 0 else None)
    plt.title(f"{img_name} - validation overlay")
    plt.legend(loc="lower right", markerscale=4)
    plt.axis("off")
    plt.savefig(os.path.join(RESULTS_DIR, f"earth_img_{number}_overlay.png"), dpi=100, bbox_inches="tight")
    plt.close()

# summary table, one row per image
with open(os.path.join(RESULTS_DIR, "summary.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["image"])
    writer.writeheader()
    writer.writerows(rows)

# averages over ALL validation images
print(f"\n===== VALIDATION RESULTS ({len(image_files)} images) =====")
print("IoU per class:")
for c, name in enumerate(CLASS_NAMES):
    print(f"  {name:10s}: {intersections[c] / unions[c]:.4f}" if unions[c] > 0 else f"  {name:10s}: not in any image")
print(f"Pixel accuracy: {correct_pixels / total_pixels:.4f}" if total_pixels else "Pixel accuracy: no images")

if all_errors:
    errors, found = np.concatenate(all_errors), np.concatenate(all_found)
    print(f"\nCoastline (thin sub-pixel line, all images together, 1 original pixel = {MRAD_PER_PIXEL:.3f} mrad):")
    print(f"  error median {np.median(errors):.2f} px ({np.median(errors) * MRAD_PER_PIXEL:.3f} mrad), "
          f"mean {errors.mean():.2f} px ({errors.mean() * MRAD_PER_PIXEL:.3f} mrad), "
          f"90% under {np.percentile(errors, 90):.2f} px ({np.percentile(errors, 90) * MRAD_PER_PIXEL:.3f} mrad)")
    print(f"  true coastline found (line within 1 model pixel): {100 * found.mean():.0f}%")
else:
    print("\nNo visible coastline in any validation image")

print(f"\nSaved overlay plots + summary.csv to {RESULTS_DIR}")
