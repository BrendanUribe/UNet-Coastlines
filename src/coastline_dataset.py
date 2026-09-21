import os #file finding/paths
from PIL import Image # open images
from torch.utils.data import Dataset # pytorch dataset class - can feed images into nn
import torchvision.transforms as T # image transformations (resizing)
from torchvision.transforms import InterpolationMode # how images are resized
import cv2 # image processing
import numpy as np # array processing
import torch

# test to prove python can pair data

# the land/water mask labels background/space the same as water (both "not land"), but the
# label-audit script showed avg water-labeled RGB is near-black (~0.04) - i.e. most of what's
# labeled "water" is actually empty space around the Earth disc, not real ocean, since space
# vastly outnumbers real ocean pixels in these renders. Training on that as "water" mostly
# teaches the model to recognize black space, diluting the signal for what real ocean looks
# like. IGNORE_INDEX marks those pixels so the loss skips them entirely instead of counting
# them as water.
IGNORE_INDEX = -100  # matches nn.CrossEntropyLoss's default ignore_index
BACKGROUND_BRIGHTNESS_THRESHOLD = 0.02  # mean RGB below this = treated as empty space, not ocean


class CoastlineDataset(Dataset): # create a dataset called coaslinedataset
    def __init__(self, image_dir, mask_dir, cloud_mask_dir, img_size=(256, 256)): # change img_size for different resolution input desired
        # inputs 
        # image dir - where the images are stored
        # mask dir - where the masks are stored
        # cloud mask dir - where the cloud masks are stored
        # img size - target image resizeing
        
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.cloud_mask_dir = cloud_mask_dir
        self.img_size = img_size

        # looks in files for .png removes any mask png and sorts
        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith(".png") and "MASK" not in f
        ])

        # steps for image transformations and image resizg help for comp cost and ensure same dims for inputs
        self.image_transform = T.Compose([
            # T.CenterCrop(512),   
            T.Resize(img_size), # img_size are the rgb images
            T.ToTensor() #nn recognizes tensors (numbers)
        ])

        # masking transformations
        self.mask_transform = T.Compose([
            # T.CenterCrop(512),   
            T.Resize(img_size, interpolation=InterpolationMode.NEAREST), # resizing mask with interpolation to ensure 0,1 and no blurr
            T.Grayscale(num_output_channels=1), # forces 1 channel for grayscale
            T.ToTensor() # tensor
        ])

    def __len__(self): # returns number of image files
        return len(self.image_files)

    def make_edge_mask(self, land_water_np):
        land_water_float = land_water_np.astype(np.float32)

        sobel_x = cv2.Sobel(land_water_float, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(land_water_float, cv2.CV_32F, 0, 1, ksize=3)

        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        edge = (gradient_magnitude > 0).astype(np.float32)
        return edge

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        number = img_name.replace("earth_img_", "").replace(".png", "")

        mask_name = f"earth_img_MASK{number}.png"
        cloud_mask_name = f"earth_img_CLOUDMASK{number}.png"

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        cloud_path = os.path.join(self.cloud_mask_dir, cloud_mask_name)

        image = Image.open(img_path).convert("RGB")
        land_water = Image.open(mask_path).convert("L")
        cloud = Image.open(cloud_path).convert("L")

        image = self.image_transform(image)
        land_water = self.mask_transform(land_water)
        cloud = self.mask_transform(cloud)

        land_water_bin = (land_water > 0.5).float()
        cloud_bin = (cloud > 0.5).float()

        seg_label = land_water_bin.squeeze(0).long()
        seg_label[cloud_bin.squeeze(0) > 0.5] = 2

        # exclude black background/space pixels from the loss - see IGNORE_INDEX comment above
        brightness = image.mean(dim=0)  # (H, W), averaged over RGB channels
        is_background = (land_water_bin.squeeze(0) == 0) & (cloud_bin.squeeze(0) == 0) & (brightness < BACKGROUND_BRIGHTNESS_THRESHOLD)
        seg_label[is_background] = IGNORE_INDEX

        lw_np = land_water_bin.squeeze(0).numpy().astype(np.uint8)
        edge_np = self.make_edge_mask(lw_np)
        edge_label = torch.from_numpy(edge_np)

        cloud_label = cloud_bin.squeeze(0)

        return image, seg_label, edge_label, cloud_label