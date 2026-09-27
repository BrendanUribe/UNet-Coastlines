# combines the 4 flat renders from gen_earth_label_renders (functions.py) into ONE label map per image
# label map = single channel png, each pixel value is a class number (not a color):
#   0 = space, 1 = water, 2 = land, 3 = cloud, 4 = dark (night side)
# priority when classes overlap: space > dark > cloud > land/water
# coastline is NOT stored, training computes it from this map (land touching water only)

import os # file finding/paths
import numpy as np # math
from PIL import Image # open/save images

RENDER_DIR = "." # where the generator saved earth_img_LANDFLAT<N>.png etc (it saves in the working dir)
LABEL_DIR = "dataset/labels" # where earth_img_LABEL<N>.png are written
PREVIEW_DIR = "dataset/label_previews" # color versions to check by eye, set to None to skip

# thresholds (0-1 brightness) - tune if labels look wrong in the previews
LAND_THRESHOLD = 0.08 # color_oceanblack map: ocean is black, anything brighter is land
CLOUD_THRESHOLD = 0.4 # cloud map coverage that counts as cloud, same as coverage_threshold in coastline_func
DARK_THRESHOLD = 0.1 # sunlit render: below this = night side / too dark to see coastline

SPACE, WATER, LAND, CLOUD, DARK = 0, 1, 2, 3, 4
PREVIEW_COLORS = np.array([
    [0, 0, 0], # space - black
    [30, 60, 200], # water - blue
    [40, 160, 60], # land - green
    [230, 230, 230], # cloud - white
    [90, 60, 110], # dark - purple
], dtype=np.uint8)


def load_gray(path): # render -> 0-1 brightness array (max over rgb so colored land counts fully)
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32).max(axis=2) / 255.0


def make_label(number):
    land = load_gray(os.path.join(RENDER_DIR, f"earth_img_LANDFLAT{number}.png")) > LAND_THRESHOLD
    cloud = load_gray(os.path.join(RENDER_DIR, f"earth_img_CLOUDFLAT{number}.png")) > CLOUD_THRESHOLD
    disk = load_gray(os.path.join(RENDER_DIR, f"earth_img_DISK{number}.png")) > 0.5
    lit = load_gray(os.path.join(RENDER_DIR, f"earth_img_LIT{number}.png")) > DARK_THRESHOLD

    label = np.full(land.shape, WATER, dtype=np.uint8) # start as water, then overwrite in reverse priority
    label[land] = LAND
    label[cloud] = CLOUD
    label[~lit] = DARK
    label[~disk] = SPACE
    return label


if __name__ == "__main__":
    os.makedirs(LABEL_DIR, exist_ok=True)
    if PREVIEW_DIR:
        os.makedirs(PREVIEW_DIR, exist_ok=True)

    # find every image number that has a DISK render
    numbers = sorted(
        f[len("earth_img_DISK"):-len(".png")]
        for f in os.listdir(RENDER_DIR)
        if f.startswith("earth_img_DISK") and f.endswith(".png")
    )
    print("Found", len(numbers), "sets of renders")

    for number in numbers:
        label = make_label(number)
        Image.fromarray(label).save(os.path.join(LABEL_DIR, f"earth_img_LABEL{number}.png"))
        if PREVIEW_DIR:
            Image.fromarray(PREVIEW_COLORS[label]).save(os.path.join(PREVIEW_DIR, f"earth_img_LABEL{number}.png"))

        fractions = np.bincount(label.ravel(), minlength=5) / label.size # class share of the image
        print(f"{number}: space {fractions[0]:.2f}, water {fractions[1]:.2f}, land {fractions[2]:.2f}, "
              f"cloud {fractions[3]:.2f}, dark {fractions[4]:.2f}")

    print("Done")
