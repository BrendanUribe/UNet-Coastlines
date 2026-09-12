import torch 
from PIL import Image # image opening
import torchvision.transforms as T # preprocessing tools for resizing and converting images to tensors
import matplotlib.pyplot as plt # plotting
import cv2 # open cv for edge detection
import numpy as np # math, ioU, Dice
import time  # timer
from UNet import UNet # unet model

print("STARTED")

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(in_channels=3, out_channels=1).to(device) # rgb to 1 channel mask 
# trained weighted model
model.load_state_dict(torch.load("unet_256_100ep_pw5_2025-05-04.pth", map_location=device)) # change .pth name (unet_<res>_<epochs>ep_pw<posweight>_<date>.pth)
model.eval() # evaluation mode for predicting not training

# Load image can change number to desired image can see where it worked well where it didnt
img_path = "dataset/images/earth_img_0.png"
image = Image.open(img_path).convert("RGB") # this one is rgb image

transform = T.Compose([
    # T.CenterCrop(512), # originally had it cropped bc some pics were far out but defeated the purpose of training on different scales 
    T.Resize((256, 256)), # esnure same size from traing
    T.ToTensor() # make tensor
])

input_tensor = transform(image).unsqueeze(0).to(device) # adda batch dim

image_plot = transform(image).permute(1, 2, 0).numpy()

# TOTAL TIMER START
total_start = time.time()

# MODEL TIMER START
model_start = time.time()

# Predict
with torch.inference_mode():
    output = model(input_tensor) # trained unet predicts mask from input image tensor gives predicted mask
    prob = torch.sigmoid(output) # sigmoid to convert logits to probabilities between 0 and 1 for binary classification (land/water)
    mask = (prob > 0.5).float() # rounds to 0 or 1

model_time = time.time() - model_start  # ⏱ model time

# Convert to numpy for plotting
mask_np = mask.squeeze().cpu().numpy()

# Load truth mask for same image
truth_path = "dataset/masks/earth_img_MASK0.png"
truth = Image.open(truth_path).convert("L")

truth_transform = T.Compose([
   # T.CenterCrop(512),
    T.Resize((256, 256)),
    T.Grayscale(num_output_channels=1),
    T.ToTensor()
])

truth_tensor = truth_transform(truth)
truth_bin = (truth_tensor.squeeze().numpy() > 0.5).astype(np.uint8) # convert to binary mask for evaluation

pred_bin = mask_np.astype(np.uint8) # convert predicted mask to binary for evaluation

# IoU 
intersection = np.logical_and(pred_bin, truth_bin).sum() # counts pixels where both predict and truth say land
union = np.logical_or(pred_bin, truth_bin).sum() # counts pixels where either prediction or truth say land
iou = intersection / union if union > 0 else 0 # intersect over union overlap/total area

# Dice measures overlap between predicted and truth
dice = (2 * intersection) / (pred_bin.sum() + truth_bin.sum()) if (pred_bin.sum() + truth_bin.sum()) > 0 else 0

print(f"IoU Score: {iou:.4f}")
print(f"Dice Score: {dice:.4f}")

# EDGE TIMER START
edge_start = time.time()

# Edge detection runs canny on predicted mask to extract boundary - boundary is PREDICTED coastline
edges = cv2.Canny((mask_np * 255).astype(np.uint8), 100, 200) # *255 for plotting bc opencv expects 0 and 255

edge_time = time.time() - edge_start  # ⏱ edge time

# TOTAL TIMER END
total_time = time.time() - total_start

# Print timing
print(f"\nModel inference time: {model_time:.4f} sec")
print(f"Edge detection time: {edge_time:.4f} sec")
print(f"Total prediction time: {total_time:.4f} sec\n")

# Plot results
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("RGB Image")
plt.imshow(image_plot)
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Predicted Mask")
plt.imshow(mask_np, cmap="gray")
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Extracted Coastline")
plt.imshow(edges, cmap="gray")
plt.axis("off")

plt.show()

error_map = np.zeros((256, 256, 3), dtype=np.uint8)

# green = correct land
error_map[(pred_bin == 1) & (truth_bin == 1)] = [0, 255, 0]

# red = false positive
error_map[(pred_bin == 1) & (truth_bin == 0)] = [255, 0, 0]

# blue = missed land
error_map[(pred_bin == 0) & (truth_bin == 1)] = [0, 0, 255]

plt.figure()
plt.title("Accuracy Map: Green correct, Red false positive, Blue missed")
plt.imshow(error_map)
plt.axis("off")
plt.show()