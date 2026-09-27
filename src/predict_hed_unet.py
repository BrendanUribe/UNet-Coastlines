import torch
from PIL import Image # image opening
import torchvision.transforms.functional as TF # image to tensor
import matplotlib.pyplot as plt # plotting
import numpy as np # math, IoU
import time # timer
from hed_unet import HEDUNet # HED-UNet model
from coastline_dataset import letterbox # same no-stretch resize + padding as training
from train_hed_unet import CLASS_NAMES, NUM_CLASSES, coastline_from_label # same classes + coastline rule as training

print("STARTED")

IMG_SIZE = 512 # must match img_size in train_hed_unet.py

# colors for plotting class maps: space, water, land, cloud, dark
CLASS_COLORS = np.array([[0, 0, 0], [30, 60, 200], [40, 160, 60], [230, 230, 230], [90, 60, 110]], dtype=np.uint8)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = HEDUNet(in_channels=3, out_channels=NUM_CLASSES + 1).to(device) # rgb to 5 classes + coastline
model.load_state_dict(torch.load("hedunet_multiclass_512_100ep.pth", map_location=device)) # must match .pth name from train_hed_unet.py
model.eval() # evaluation mode for predicting not training

# Load image can change number to desired image
number = "23"
img_path = f"dataset/images/earth_img_{number}.png"
label_path = f"dataset/labels/earth_img_LABEL{number}.png"
image_full = Image.open(img_path).convert("RGB") # original full resolution image
label_full = Image.open(label_path) # original full resolution labels

# shrink without stretching + pad to square, same as training
image_small, (scale, pad_x, pad_y) = letterbox(image_full, IMG_SIZE, Image.BILINEAR)
label_small, _ = letterbox(label_full, IMG_SIZE, Image.NEAREST)

input_tensor = TF.to_tensor(image_small).unsqueeze(0).to(device) # add a batch dim
image_plot = np.array(image_small)

# MODEL TIMER START
model_start = time.time()

# Predict
with torch.inference_mode():
    combined, _ = model(input_tensor) # only need the merged output, per-level outputs are for training
    class_map = combined[0, :NUM_CLASSES].argmax(dim=0).cpu().numpy() # most likely class per pixel
    edge_prob = torch.sigmoid(combined[0, NUM_CLASSES]).cpu().numpy() # coastline probability
    edge_np = (edge_prob > 0.5).astype(np.uint8) # coastline predicted directly by the model

model_time = time.time() - model_start

# truth at model size (for class scores) and at FULL resolution (for the overlay)
truth = np.array(label_small)
truth_edge = coastline_from_label(torch.from_numpy(truth.astype(np.int64))[None])[0, 0].numpy()
truth_full = np.array(label_full)
truth_edge_full = coastline_from_label(torch.from_numpy(truth_full.astype(np.int64))[None])[0, 0].numpy()

# only score the real image area, not the black padding added by letterbox
valid = np.zeros_like(truth, dtype=bool)
new_w, new_h = round(image_full.width / scale), round(image_full.height / scale)
valid[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = True

# IoU per class - overlap / total area for each class
print("IoU per class:")
for c, name in enumerate(CLASS_NAMES):
    pred_c, truth_c = (class_map == c) & valid, (truth == c) & valid
    union = np.logical_or(pred_c, truth_c).sum()
    if union > 0:
        print(f"  {name:6s}: {np.logical_and(pred_c, truth_c).sum() / union:.4f}")
    else:
        print(f"  {name:6s}: not in image")

print(f"\nPixel accuracy: {(class_map == truth)[valid].mean():.4f}")
print(f"\nModel inference time (classes + coastline): {model_time:.4f} sec")
print(f"Each model pixel = {scale:.1f} x {scale:.1f} original pixels\n")

# map predicted coastline pixels back to ORIGINAL image pixel coordinates (undo padding + shrink)
ys, xs = np.nonzero(edge_np)
xs_full = (xs - pad_x + 0.5) * scale - 0.5
ys_full = (ys - pad_y + 0.5) * scale - 0.5
ys_true, xs_true = np.nonzero(truth_edge_full) # true coastline at full resolution

# Plot 1 - model view (256 x 256)
plt.figure(figsize=(15, 9))
plots = [
    ("Model Input (no stretch, padded)", image_plot, {}),
    ("True Classes", CLASS_COLORS[truth], {}),
    ("Predicted Classes", CLASS_COLORS[class_map], {}),
    ("True Coastline", truth_edge, dict(cmap="gray")),
    #("Coastline Probability", edge_prob, dict(cmap="gray", vmin=0, vmax=1)),
    #("Predicted Coastline", edge_np, dict(cmap="gray")),
]
for i, (title, img, kwargs) in enumerate(plots):
    plt.subplot(2, 3, i + 1)
    plt.title(title)
    plt.imshow(img, **kwargs)
    plt.axis("off")
plt.suptitle("space = black, water = blue, land = green, cloud = white, dark = purple")

# Plot 2 - coastlines overlaid on the ORIGINAL full resolution image
plt.figure(figsize=(12, 9))
plt.imshow(np.array(image_full))
plt.scatter(xs_full, ys_full, s=4, c="red", marker="s", linewidths=0, label="predicted coastline (model pixels)")
plt.scatter(xs_true, ys_true, s=0.3, c="lime", linewidths=0, label="true coastline (full resolution)")
plt.title(f"Coastline overlay on original image ({image_full.width} x {image_full.height})")
plt.legend(loc="lower right", markerscale=4)
plt.axis("off")

plt.show()
