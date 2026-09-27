import torch
import torch.nn.functional as F # cross entropy, pooling, log sigmoid
from torch.utils.data import DataLoader # takes dataset and feeds to model batches
from coastline_dataset import CoastlineLabelDataset # rgb images + multi-class label maps
from hed_unet import HEDUNet # imports HED-UNet model
import torch.optim as optim # optimizers
import time # timer

STACK_HEIGHT = 5 # number of down/up levels in HED-UNet, img_size must be divisible by 2**STACK_HEIGHT

# classes in the label maps (from make_label_masks.py)
CLASS_NAMES = ["space", "water", "land", "cloud", "dark"]
NUM_CLASSES = len(CLASS_NAMES)
WATER, LAND = 1, 2
# model output channels: 0-4 = class scores, 5 = coastline edge


def coastline_from_label(label): # label [B,H,W] -> coastline [B,1,H,W] of 0/1
    # coastline = land pixel touching water OR water pixel touching land
    # cloud/dark/space boundaries are never coastline
    land = (label == LAND).float().unsqueeze(1)
    water = (label == WATER).float().unsqueeze(1)
    near_land = F.max_pool2d(land, 3, stride=1, padding=1) # 1 if any land in 3x3 neighborhood
    near_water = F.max_pool2d(water, 3, stride=1, padding=1)
    return ((land * near_water) + (water * near_land)).clamp(max=1)


def get_targets(label): # builds (class map, coastline) targets at every scale the model predicts at
    with torch.no_grad():
        edge = coastline_from_label(label)
        targets = []
        for level in range(STACK_HEIGHT + 1):
            s = 2 ** level
            level_label = label[:, ::s, ::s] # nearest downsample keeps class numbers
            level_edge = F.max_pool2d(edge, s) if s > 1 else edge # keep thin coastline visible at coarse levels
            targets.append((level_label, level_edge))
    return targets # full size first, same order as model's level outputs


def coastline_bce(logits, target, max_weight=10.0): # BCE that boosts rare coastline pixels, capped
    # coastline is ~1-2% of pixels, so missed coastline pixels are weighted up to max_weight x more
    # (full balancing = ~65x made the model mark almost the whole Earth as coastline, cap keeps it thin)
    with torch.no_grad():
        beta = target.mean(dim=[2, 3], keepdim=True).clamp(min=1e-6) # fraction of coastline pixels per image
        pos_weight = ((1 - beta) / beta).clamp(max=max_weight)
    return (-pos_weight * target * F.logsigmoid(logits)
            - (1 - target) * F.logsigmoid(-logits)).mean()


def hed_loss(prediction, target): # class loss + coastline loss for one scale
    label, edge = target
    class_loss = F.cross_entropy(prediction[:, :NUM_CLASSES], label)
    edge_loss = coastline_bce(prediction[:, NUM_CLASSES:], edge)
    return class_loss + edge_loss


if __name__ == "__main__":
    print("SCRIPT STARTED")

    # Dataset - rgb images + label maps
    dataset = CoastlineLabelDataset(
        image_dir="dataset/images",
        label_dir="dataset/labels",
        img_size=512 # square side fed to model, make sure this matches prediction (IMG_SIZE in predict_hed_unet.py)
    )
    # loader batch can change rn 4 images at a time and shuffle for randomly mixed each epoch
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )
    print("Number of training images:", len(dataset))

    # Model - cuda for nvidia not amd
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # creates HED-UNet model 3 channel rgb to 5 class scores + 1 coastline channel
    model = HEDUNet(in_channels=3, out_channels=NUM_CLASSES + 1, stack_height=STACK_HEIGHT).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3) # HED-UNet default lr

    # Training loop
    epochs = 100 # model passes thru all training images this many times - can change

    #  TOTAL TIMER START
    total_start_time = time.time()

    for epoch in range(epochs): # training loop for each epoch
        model.train()
        epoch_start_time = time.time() # timer for each epoch
        total_loss = 0 # loss initialize

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            combined, levels = model(images)
            targets = get_targets(labels)

            # loss = final merged output + every level's output (deep supervision)
            loss = hed_loss(combined, targets[0])
            for level_pred, level_target in zip(levels, targets):
                loss = loss + hed_loss(level_pred, level_target)

            optimizer.zero_grad() # clearing old gradient info before new updates
            loss.backward() # calculate each model weight contribution to error (math step)
            optimizer.step() # updates model weight (learning)

            total_loss += loss.item() # accumulate loss for epoch

        epoch_time = time.time() - epoch_start_time # timer for epoch
        avg_loss = total_loss / len(loader) # avg loss across batches

        print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}, Time: {epoch_time:.2f} sec") # epoch number, loss, time

        # checkpoint saves for epochs every 10
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"hed_checkpoint_epoch_{epoch+1}.pth")
            print(f"Checkpoint saved at epoch {epoch+1}")

    # Save model
    torch.save(model.state_dict(), "hedunet_multiclass_512_100ep.pth") # change .pth name (hedunet_multiclass_<res>_<epochs>ep_<date>.pth)
    print("Model saved")

    # TOTAL TIMER END
    total_time = time.time() - total_start_time

    print(f"\nTotal training time: {total_time:.2f} seconds")
    print(f"Total training time: {total_time/60:.2f} minutes")
