import torch
from PIL import Image # image opening
import torchvision.transforms as T # preprocessing tools for resizing and converting images to tensors
import matplotlib.pyplot as plt # plotting
import numpy as np # math, IoU
import time # timer
from hed_unet import HEDUNet # HED-UNet model
from train_hed_unet import CLASS_NAMES, NUM_CLASSES, coastline_from_label # same classes + coastline rule as training

print("STARTED")

# colors for plotting class maps: space, water, land, cloud, dark
CLASS_COLORS = np.array([[0, 0, 0], [30, 60, 200], [40, 160, 60], [230, 230, 230], [90, 60, 110]], dtype=np.uint8)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = HEDUNet(in_channels=3, out_channels=NUM_CLASSES + 1).to(device) # rgb to 5 classes + coastline
model.load_state_dict(torch.load("hedunet_multiclass_256_100ep.pth", map_location=device)) # must match .pth name from train_hed_unet.py
model.eval() # evaluation mode for predicting not training

# Load image can change number to desired image
number = "23"
img_path = f"dataset/images/earth_img_{number}.png"
label_path = f"dataset/labels/earth_img_LABEL{number}.png"
image = Image.open(img_path).convert("RGB")

transform = T.Compose([
    T.Resize((256, 256)), # ensure same size from training
    T.ToTensor() # make tensor
])

input_tensor = transform(image).unsqueeze(0).to(device) # add a batch dim
image_plot = transform(image).permute(1, 2, 0).numpy()

# MODEL TIMER START
model_start = time.time()

# Predict
with torch.inference_mode():
    combined, _ = model(input_tensor) # only need the merged output, per-level outputs are for training
    class_map = combined[0, :NUM_CLASSES].argmax(dim=0).cpu().numpy() # most likely class per pixel
    edge_prob = torch.sigmoid(combined[0, NUM_CLASSES]).cpu().numpy() # coastline probability
    edge_np = (edge_prob > 0.5).astype(np.uint8) # coastline predicted directly by the model

model_time = time.time() - model_start

# Load truth label map for same image (nearest resize keeps class numbers)
truth = np.array(T.Resize((256, 256), interpolation=T.InterpolationMode.NEAREST)(Image.open(label_path)))
truth_edge = coastline_from_label(torch.from_numpy(truth.astype(np.int64))[None])[0, 0].numpy()

# IoU per class - overlap / total area for each class
print("IoU per class:")
for c, name in enumerate(CLASS_NAMES):
    pred_c, truth_c = class_map == c, truth == c
    union = np.logical_or(pred_c, truth_c).sum()
    if union > 0:
        print(f"  {name:6s}: {np.logical_and(pred_c, truth_c).sum() / union:.4f}")
    else:
        print(f"  {name:6s}: not in image")

print(f"\nPixel accuracy: {(class_map == truth).mean():.4f}")
print(f"\nModel inference time (classes + coastline): {model_time:.4f} sec\n")

# Plot results
plt.figure(figsize=(18, 7))

plots = [
    ("RGB Image", image_plot, {}),
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
plt.show()
