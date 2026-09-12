import os #file finding/paths
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