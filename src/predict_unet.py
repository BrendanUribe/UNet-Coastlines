import torch 
from PIL import Image # image opening
import torchvision.transforms as T # preprocessing tools for resizing and converting images to tensors
import matplotlib.pyplot as plt # plotting
import cv2 # open cv for edge detection
import numpy as np # math, ioU, Dice
import time  # timer
from UNet import UNet # unet model
from matplotlib import cm


def non_max_suppress_thin(edge_prob):
    """Keeps only the local maxima in the edge probability map, thinning the edges to a single pixel width."""
    gx = cv2.Sobel(edge_prob, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(edge_prob, cv2.CV_64F, 0, 1, ksize=3)
    angle = (np.arctan2(gy, gx) * 180.0 / np.pi) % 180.0

    H, W = edge_prob.shape
    thinned = np.zeros_like(edge_prob)

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            a = angle[i, j]
            if a < 22.5 or a >= 157.5:
                n1, n2 = edge_prob[i, j - 1], edge_prob[i, j + 1]
            elif a < 67.5:
                n1, n2 = edge_prob[i - 1, j + 1], edge_prob[i + 1, j - 1]
            elif a < 112.5:
                n1, n2 = edge_prob[i - 1, j], edge_prob[i + 1, j]
            else:
                n1, n2 = edge_prob[i - 1, j - 1], edge_prob[i + 1, j + 1]

            if edge_prob[i, j] >= n1 and edge_prob[i, j] >= n2:
                thinned[i, j] = edge_prob[i, j]

    return thinned

print("STARTED")

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(in_channels=3, num_classes=3).to(device) # rgb to 3 channel mask 
# trained weighted model
model.load_state_dict(torch.load("unet_256_100ep_2headed_2025-XX-XX.pth", map_location=device)) # change .pth name (unet_<res>_<epochs>ep_pw<posweight>_<date>.pth)
model.eval() # evaluation mode for predicting not training

# Load image can change number to desired image can see where it worked well where it didnt
img_path = "dataset/images/earth_img_3.png"
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
    seg_out, edge_outputs = model(input_tensor)

    print("edge_out min/max (raw logits):", edge_outputs[-1].min().item(), edge_outputs[-1].max().item())
    print("seg_out per-class mean score:", seg_out.mean(dim=[0,2,3]))

    seg_pred = torch.argmax(seg_out, dim=1).squeeze().cpu().numpy()

    edge_prob = torch.sigmoid(edge_outputs[-1]).squeeze().cpu().numpy()   # stays as continuous 0-1 values, no threshold yet

model_time = time.time() - model_start

# Suppress edge predictions that fall inside predicted cloud regions, BEFORE thinning
cloud_pixels = (seg_pred == 2)
edge_prob[cloud_pixels] = 0

background_pixels = (image_plot.mean(axis=2) < 0.02)  # very dark pixels, background/space
norm = plt.Normalize(vmin=0, vmax=2)
seg_display = cm.viridis(norm(seg_pred))
seg_display[background_pixels] = [0.12, 0.12, 0.12, 1.0]   # force background to neutral gray
edge_prob[background_pixels] = 0

edge_thin_prob = non_max_suppress_thin(edge_prob)   # NMS runs on the continuous probability map
edge_thin = (edge_thin_prob > 0.5).astype(np.float32)   # threshold happens LAST

total_time = time.time() - total_start

print(f"\nModel inference time: {model_time:.4f} sec")
print(f"Total prediction time: {total_time:.4f} sec\n")

# build an RGB overlay: the original image with predicted coastline pixels painted bright red
coastline_overlay = image_plot.copy()
coastline_overlay[edge_thin > 0] = [1.0, 0.0, 0.0]  # red for coastline

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
plt.title("Predicted Coastline (thin)")
plt.imshow(edge_thin, cmap="gray")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.title("Coastline over RGB")
plt.imshow(coastline_overlay)
plt.axis("off")



plt.show()