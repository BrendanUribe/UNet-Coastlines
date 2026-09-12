import torch
from torch.utils.data import DataLoader # takes dataset and feeds to model batches
from coastline_dataset import CoastlineDataset # imports coastline data from rgb and mask pairs
from UNet import UNet # imports unet model
import torch.nn as nn # neural network tools
import torch.optim as optim # # optimizers 
import time # timer

print("SCRIPT STARTED")

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

# creates Unet model 3 channel rgb to 1 channel mask
model = UNet(in_channels=3, out_channels=1).to(device)

# Loss + optimizer
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(device)) # BCE for binary classification and weight 
optimizer = optim.Adam(model.parameters(), lr=5e-5) # updates model weights adam is common  lr for learning rate small rate = slower but more stable

# Training loop
epochs = 100 # model passes thru all training images this many times - can change

#  TOTAL TIMER START
total_start_time = time.time()

for epoch in range(epochs): # training loop for each epoch
    epoch_start_time = time.time() # timer for each epoch
    total_loss = 0 # loss initialize

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks) # compares predicted mask vs true result is error/loss

        optimizer.zero_grad() # clearing old gradient info before new updates 
        loss.backward() # calculate each model weight contribution to error (math step)
        optimizer.step() # updates model weight (learning)

        total_loss += loss.item() # accumulate loss for epoch

    epoch_time = time.time() - epoch_start_time # timer for epoch
    avg_loss = total_loss / len(loader) # avg loss across batches

    print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}, Time: {epoch_time:.2f} sec") # epoch number, loss, time

    # checkpoint saves for epochs every 10 bc I accidentally cancelled it before :(
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")
        print(f"Checkpoint saved at epoch {epoch+1}")

# Save model
torch.save(model.state_dict(), "unet_256_100ep_pw5_2025-05-04.pth") # change .pth name (unet_<res>_<epochs>ep_pw<posweight>_<date>.pth)
print("Model saved")

# TOTAL TIMER END
total_time = time.time() - total_start_time

print(f"\nTotal training time: {total_time:.2f} seconds")
print(f"Total training time: {total_time/60:.2f} minutes")