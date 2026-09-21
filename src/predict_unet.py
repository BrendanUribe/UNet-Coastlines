import torch
from PIL import Image # image opening
import torchvision.transforms as T # preprocessing tools for resizing and converting images to tensors
import matplotlib.pyplot as plt # plotting
from matplotlib import cm # for manually compositing the segmentation colormap
import cv2 # open cv for edge detection
import numpy as np # math, ioU, Dice
from skimage.morphology import skeletonize # thins a blobby binary mask down to a 1px-wide centerline
import time  # timer
from UNet import UNet # unet model

# must match coastline_dataset.py's BACKGROUND_BRIGHTNESS_THRESHOLD - the model gets zero
# training signal on these pixels (see IGNORE_INDEX), so its prediction there is arbitrary
# noise, not a real answer. Recompute the same mask directly from the image at inference
# time instead of trusting whatever class the model happened to guess.
BACKGROUND_BRIGHTNESS_THRESHOLD = 0.02

print("STARTED")

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(in_channels=3, num_classes=3).to(device) # rgb to 3 channel mask 
# trained weighted model
model.load_state_dict(torch.load("unet_256_100ep_2headed_2025-XX-XX.pth", map_location=device)) # change .pth name (unet_<res>_<epochs>ep_pw<posweight>_<date>.pth)
model.eval() # evaluation mode for predicting not training

# Load image can change number to desired image can see where it worked well where it didnt
img_path = "dataset/images/earth_img_2.png"
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
    seg_out, edge_out = model(input_tensor)

    print("edge_out min/max (raw logits):", edge_out.min().item(), edge_out.max().item())
    print("seg_out per-class mean score:", seg_out.mean(dim=[0,2,3]))

    seg_pred = torch.argmax(seg_out, dim=1).squeeze().cpu().numpy()  # 0=water,1=land,2=cloud per pixel

    edge_prob = torch.sigmoid(edge_out)
    edge_mask = (edge_prob > 0.5).float().squeeze().cpu().numpy()

model_time = time.time() - model_start

# background pixels (empty space around the Earth disc) were excluded from training via
# IGNORE_INDEX, so the model was never taught what to output there - its raw prediction on
# them is arbitrary and was bleeding fake "coastline" into the edge output. Recompute the
# same near-black brightness test used during training, directly from the input image.
background_pixels = (image_plot.mean(axis=2) < BACKGROUND_BRIGHTNESS_THRESHOLD)

# Suppress edge predictions that fall inside predicted cloud regions OR background space
cloud_pixels = (seg_pred == 2)
edge_mask_clean = edge_mask.copy()
edge_mask_clean[cloud_pixels] = 0
edge_mask_clean[background_pixels] = 0

# the raw thresholded edge mask is a blobby region wherever edge_prob > 0.5, not a clean
# boundary - skeletonize collapses it down to a 1-pixel-wide centerline so it actually reads
# as a coastline instead of a filled mask
edge_thin = skeletonize(edge_mask_clean.astype(bool)).astype(np.float32)

total_time = time.time() - total_start

print(f"\nModel inference time: {model_time:.4f} sec")
print(f"Total prediction time: {total_time:.4f} sec\n")

# build a segmentation display where background pixels are forced to a neutral dark gray,
# regardless of whatever (untrained, meaningless) class the model guessed there
norm = plt.Normalize(vmin=0, vmax=2)
seg_display = cm.viridis(norm(seg_pred))
seg_display[background_pixels] = [0.12, 0.12, 0.12, 1.0]

# build an RGB overlay: the original image with the thin predicted coastline painted bright red
coastline_overlay = image_plot.copy()
coastline_overlay[edge_thin > 0] = [1.0, 0.0, 0.0]

# Plot results
plt.figure(figsize=(16, 4))

plt.subplot(1, 4, 1)
plt.title("RGB Image")
plt.imshow(image_plot)
plt.axis("off")

plt.subplot(1, 4, 2)
plt.title("Predicted Land/Water/Cloud")
plt.imshow(seg_display)
plt.axis("off")

plt.subplot(1, 4, 3)
plt.title("Predicted Coastline (thin, edge head)")
plt.imshow(edge_thin, cmap="gray")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.title("Coastline over RGB")
plt.imshow(coastline_overlay)
plt.axis("off")



plt.show()