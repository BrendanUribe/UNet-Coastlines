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

# Suppress edge predictions that fall inside predicted cloud regions
cloud_pixels = (seg_pred == 2)
edge_mask_clean = edge_mask.copy()
edge_mask_clean[cloud_pixels] = 0

total_time = time.time() - total_start

print(f"\nModel inference time: {model_time:.4f} sec")
print(f"Total prediction time: {total_time:.4f} sec\n")

# Plot results
plt.figure(figsize=(16, 4))

plt.subplot(1, 4, 1)
plt.title("RGB Image")
plt.imshow(image_plot)
plt.axis("off")

plt.subplot(1, 4, 2)
plt.title("Predicted Land/Water/Cloud")
plt.imshow(seg_pred, cmap="viridis", vmin=0, vmax=2)
plt.axis("off")



plt.show()