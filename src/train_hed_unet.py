import torch
import torch.nn.functional as F # sobel conv, pooling, log sigmoid
from torch.utils.data import DataLoader # takes dataset and feeds to model batches
from coastline_dataset import CoastlineDataset # same rgb and mask pairs as the UNet
from hed_unet import HEDUNet # imports HED-UNet model
import torch.optim as optim # optimizers
import time # timer

print("SCRIPT STARTED")

STACK_HEIGHT = 5 # number of down/up levels in HED-UNet, img_size must be divisible by 2**STACK_HEIGHT


# Sobel filters to find where the mask changes (land <-> water) = coastline
SOBEL = torch.tensor([
    [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], # horizontal change
    [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], # vertical change
], dtype=torch.float32).reshape(2, 1, 3, 3)


def mask_to_edge(mask): # mask [B,1,H,W] -> coastline edge [B,1,H,W] of 0/1
    padded = F.pad(mask, (1, 1, 1, 1), mode='replicate') # replicate so image border isn't counted as coastline
    grad = F.conv2d(padded, SOBEL.to(mask.device))
    return (grad != 0).any(dim=1, keepdim=True).float()


def get_targets(mask): # builds [mask, edge] targets at every scale the model predicts at
    with torch.no_grad():
        targets = []
        for _ in range(STACK_HEIGHT + 1):
            targets.append(torch.cat([mask, mask_to_edge(mask)], dim=1)) # channel 0 = mask, channel 1 = edge
            mask = F.avg_pool2d(mask, 2) # half size for next (coarser) level
    return targets # full size first, same order as model's level outputs


def auto_weight_bce(logits, target): # BCE that balances classes per image (coastline pixels are rare)
    with torch.no_grad():
        beta = target.mean(dim=[2, 3], keepdim=True) # fraction of positive pixels per image & channel
    return (-(1 - beta) * target * F.logsigmoid(logits)
            - beta * (1 - target) * F.logsigmoid(-logits)).mean()


# Dataset - creates where data set is and size
dataset = CoastlineDataset(
    image_dir="dataset/images",
    mask_dir="dataset/masks",
    img_size=(256, 256) # make sure this matches across dataset, training, prediction
)
# loader batch can change rn 4 images at a time and shuffle for randomly mixed each epoch
loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

# Model - cuda for nvidia not amd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# creates HED-UNet model 3 channel rgb to 2 channels (mask + coastline edge)
model = HEDUNet(in_channels=3, out_channels=2, stack_height=STACK_HEIGHT).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3) # HED-UNet default lr

# Training loop
epochs = 100 # model passes thru all training images this many times - can change

#  TOTAL TIMER START
total_start_time = time.time()

for epoch in range(epochs): # training loop for each epoch
    model.train()
    epoch_start_time = time.time() # timer for each epoch
    total_loss = 0 # loss initialize

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        combined, levels = model(images)
        targets = get_targets(masks)

        # loss = final merged output + every level's output (deep supervision)
        loss = auto_weight_bce(combined, targets[0])
        for level_pred, level_target in zip(levels, targets):
            loss = loss + auto_weight_bce(level_pred, level_target)

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
torch.save(model.state_dict(), "hedunet_256_100ep.pth") # change .pth name (hedunet_<res>_<epochs>ep_<date>.pth)
print("Model saved")

# TOTAL TIMER END
total_time = time.time() - total_start_time

print(f"\nTotal training time: {total_time:.2f} seconds")
print(f"Total training time: {total_time/60:.2f} minutes")
