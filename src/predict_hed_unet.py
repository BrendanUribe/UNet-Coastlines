import torch
from PIL import Image # image opening
import torchvision.transforms as T # preprocessing tools for resizing and converting images to tensors
import matplotlib.pyplot as plt # plotting
import numpy as np # math, IoU, Dice
import time # timer
from hed_unet import HEDUNet # HED-UNet model

print("STARTED")

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = HEDUNet(in_channels=3, out_channels=2).to(device) # rgb to mask + coastline edge
model.load_state_dict(torch.load("hedunet_256_100ep.pth", map_location=device)) # must match .pth name from train_hed_unet.py
model.eval() # evaluation mode for predicting not training

# Load image can change number to desired image
img_path = "dataset/images/earth_img_0.png"
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
    prob = torch.sigmoid(combined)[0] # [2, H, W] probabilities
    mask_np = (prob[0] > 0.5).float().cpu().numpy() # channel 0 = land/water mask
    edge_prob = prob[1].cpu().numpy() # channel 1 = coastline probability
    edge_np = (prob[1] > 0.5).float().cpu().numpy() # coastline predicted directly by the model (no Canny)

model_time = time.time() - model_start

# Load truth mask for same image
truth_path = "dataset/masks/earth_img_MASK0.png"
truth = Image.open(truth_path).convert("L")

truth_transform = T.Compose([
    T.Resize((256, 256), interpolation=T.InterpolationMode.NEAREST),
    T.Grayscale(num_output_channels=1),
    T.ToTensor()
])

truth_bin = (truth_transform(truth).squeeze().numpy() > 0.5).astype(np.uint8)
pred_bin = mask_np.astype(np.uint8)

# IoU and Dice on the mask - same as predict_unet.py so results are comparable
intersection = np.logical_and(pred_bin, truth_bin).sum()
union = np.logical_or(pred_bin, truth_bin).sum()
iou = intersection / union if union > 0 else 0
dice = (2 * intersection) / (pred_bin.sum() + truth_bin.sum()) if (pred_bin.sum() + truth_bin.sum()) > 0 else 0

print(f"IoU Score: {iou:.4f}")
print(f"Dice Score: {dice:.4f}")
print(f"\nModel inference time (mask + coastline): {model_time:.4f} sec\n")

# Plot results
plt.figure(figsize=(16, 4))

plt.subplot(1, 4, 1)
plt.title("RGB Image")
plt.imshow(image_plot)
plt.axis("off")

plt.subplot(1, 4, 2)
plt.title("Predicted Mask")
plt.imshow(mask_np, cmap="gray")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.title("Coastline Probability")
plt.imshow(edge_prob, cmap="gray", vmin=0, vmax=1)
plt.axis("off")

plt.subplot(1, 4, 4)
plt.title("Predicted Coastline")
plt.imshow(edge_np, cmap="gray")
plt.axis("off")

plt.show()
