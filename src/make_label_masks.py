# combines the 4 flat renders from gen_earth_label_renders (functions.py) into ONE label map per image
# label map = single channel png, each pixel value is a class number (not a color):
#   0 = space, 1 = water, 2 = land, 3 = cloud, 4 = dark (night side)
# priority when classes overlap: space > dark > cloud > land/water
# coastline is NOT stored, training computes it from this map (land touching water only)

import os # file finding/paths
import sys # command line option (train / validation)
import shutil # copy rgb images into the dataset folders
import numpy as np # math
from PIL import Image # open/save images

# run:  python src/make_label_masks.py              -> training set   (dataset/images, dataset/labels)
#       python src/make_label_masks.py validation   -> validation set (dataset/validation_images, dataset/validation_labels)
# it processes EVERY render set in RENDER_DIR, so move old renders out before generating a new set
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "train"
RENDER_DIR = "." # where the generator saved earth_img_LANDFLAT<N>.png etc (it saves in the working dir)
if SPLIT == "validation":
    IMAGE_DIR = "dataset/validation_images" # rgb images copied here
    LABEL_DIR = "dataset/validation_labels" # earth_img_LABEL<N>.png written here
    PREVIEW_DIR = "dataset/validation_label_previews" # color versions to check by eye, set to None to skip
else:
    IMAGE_DIR = "dataset/images"
    LABEL_DIR = "dataset/labels"
    PREVIEW_DIR = "dataset/label_previews"

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
    if SPLIT not in ("train", "validation"):
        sys.exit(f"unknown option '{SPLIT}', use: python src/make_label_masks.py [validation]")
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(LABEL_DIR, exist_ok=True)
    if PREVIEW_DIR:
        os.makedirs(PREVIEW_DIR, exist_ok=True)

    # find every image number that has a DISK render
    numbers = sorted(
        f[len("earth_img_DISK"):-len(".png")]
        for f in os.listdir(RENDER_DIR)
        if f.startswith("earth_img_DISK") and f.endswith(".png")
    )
    print(f"Found {len(numbers)} sets of renders -> {SPLIT} set ({IMAGE_DIR}, {LABEL_DIR})")

    skipped = [] # views with a missing render (usually a missing cloud map for that date, so POV-Ray failed)
    for number in numbers:
        needed = [f"earth_img_{number}.png"] + [f"earth_img_{kind}{number}.png" for kind in ("LANDFLAT", "CLOUDFLAT", "DISK", "LIT")]
        missing = [f for f in needed if not os.path.exists(os.path.join(RENDER_DIR, f))]
        if missing:
            skipped.append(number)
            print(f"{number}: SKIPPED, missing {', '.join(missing)}")
            continue
        label = make_label(number)
        Image.fromarray(label).save(os.path.join(LABEL_DIR, f"earth_img_LABEL{number}.png"))
        shutil.copy(os.path.join(RENDER_DIR, f"earth_img_{number}.png"), os.path.join(IMAGE_DIR, f"earth_img_{number}.png"))
        if PREVIEW_DIR:
            Image.fromarray(PREVIEW_COLORS[label]).save(os.path.join(PREVIEW_DIR, f"earth_img_LABEL{number}.png"))

        fractions = np.bincount(label.ravel(), minlength=5) / label.size # class share of the image
        print(f"{number}: space {fractions[0]:.2f}, water {fractions[1]:.2f}, land {fractions[2]:.2f}, "
              f"cloud {fractions[3]:.2f}, dark {fractions[4]:.2f}")

    print(f"Done: {len(numbers) - len(skipped)} labeled, {len(skipped)} skipped")
    if skipped:
        print("Skipped views (render failed, usually a missing cloud map for that date):", ", ".join(skipped))
