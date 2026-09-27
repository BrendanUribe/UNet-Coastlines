import os #file finding/paths
import numpy as np # label arrays
import torch # label tensors
from PIL import Image # open images
from torch.utils.data import Dataset # pytorch dataset class - can feed images into nn
import torchvision.transforms as T # image transformations (resizing)
from torchvision.transforms import InterpolationMode # how images are resized

# test to prove python can pair data 


class CoastlineDataset(Dataset): # create a dataset called coaslinedataset
    def __init__(self, image_dir, mask_dir, img_size=(256, 256)): # change img_size for different resolution input desired
        # inputs 
        # image dir - where the images are stored
        # mask dir - where the masks are stored
        # img size - target image resizeing
        
        self.image_dir = image_dir
        self.mask_dir = mask_dir
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

    def __getitem__(self, idx): # defines how to load one image and mask 
        img_name = self.image_files[idx]

        number = img_name.replace("earth_img_", "").replace(".png", "")
        mask_name = f"earth_img_MASK{number}.png"

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        mask = (mask > 0.5).float()

        return image, mask # returning image and ground trth mask

def letterbox(img, size, resample): # shrink keeping shape (no stretching), then pad to a size x size square with black
    # returns padded image + (scale, pad_x, pad_y) so pixels can be mapped back to the original image:
    #   x_original = (x_small - pad_x + 0.5) * scale - 0.5, same for y
    w, h = img.size
    scale = max(w, h) / size # original pixels per small pixel, same in x and y
    new_w, new_h = round(w / scale), round(h / scale)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas = Image.new(img.mode, (size, size), 0) # 0 = black = space class
    canvas.paste(img.resize((new_w, new_h), resample), (pad_x, pad_y))
    return canvas, (scale, pad_x, pad_y)


# multi-class version for HED-UNet: pairs earth_img_<N>.png with earth_img_LABEL<N>.png
# label values: 0 space, 1 water, 2 land, 3 cloud, 4 dark (made by make_label_masks.py)
# images are letterboxed (shrunk without stretching + padded with space) so geometry stays correct
class CoastlineLabelDataset(Dataset):
    def __init__(self, image_dir, label_dir, img_size=256): # img_size = side of the square fed to the model
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.img_size = img_size

        # only plain images earth_img_<N>.png, skips MASK/LABEL/render files if they share the folder
        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.startswith("earth_img_") and f.endswith(".png")
            and f[len("earth_img_"):-len(".png")].replace("_", "").isdigit()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        number = img_name.replace("earth_img_", "").replace(".png", "")

        image = Image.open(os.path.join(self.image_dir, img_name)).convert("RGB")
        label = Image.open(os.path.join(self.label_dir, f"earth_img_LABEL{number}.png")) # keep raw class numbers, no convert

        image, _ = letterbox(image, self.img_size, Image.BILINEAR)
        label, _ = letterbox(label, self.img_size, Image.NEAREST) # nearest so class numbers never get blended

        image = T.functional.to_tensor(image)
        label = torch.from_numpy(np.array(label, dtype=np.int64)) # [H, W] class numbers

        return image, label
