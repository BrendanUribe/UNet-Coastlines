import torch
from torch.utils.data import DataLoader # takes dataset and feeds to model batches
from coastline_dataset import CoastlineDataset # imports coastline data from rgb and mask pairs
from UNet import UNet # imports unet model
import torch.nn as nn # neural network tools
import torch.optim as optim # # optimizers 
import time # timer
from losses import combined_loss

print("SCRIPT STARTED")

# Dataset - creates where data set is and size
dataset = CoastlineDataset(
    image_dir="dataset/images",
    mask_dir="dataset/masks",
    cloud_mask_dir="dataset/cloud masks",
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

# creates Unet model 3 channel rgb 
model = UNet(in_channels=3, num_classes=3).to(device) # changed output classes to 3

# Loss + optimizer ( commented our criterion since we have total loss now)
# criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(device)) # BCE for binary classification and weight 
optimizer = optim.Adam(model.parameters(), lr=5e-5) # updates model weights adam is common  lr for learning rate small rate = slower but more stable

# Training loop
epochs = 50 # model passes thru all training images this many times - can change

#  TOTAL TIMER START
total_start_time = time.time()

for epoch in range(epochs):
    epoch_start_time = time.time()
    total_loss_sum = 0
    seg_loss_sum = 0
    edge_loss_sum = 0

    for images, seg_labels, edge_labels, cloud_labels in loader:
        images = images.to(device)
        seg_labels = seg_labels.to(device)
        edge_labels = edge_labels.to(device)
        cloud_labels = cloud_labels.to(device)

        seg_out, edge_outputs = model(images)

        total_loss, seg_loss, edge_loss = combined_loss(
            seg_out, edge_outputs, seg_labels, edge_labels, cloud_labels
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        total_loss_sum += total_loss.item()
        seg_loss_sum += seg_loss.item()
        edge_loss_sum += edge_loss.item()

    epoch_time = time.time() - epoch_start_time
    avg_total = total_loss_sum / len(loader)
    avg_seg = seg_loss_sum / len(loader)
    avg_edge = edge_loss_sum / len(loader)

    print(f"Epoch {epoch+1}, Total Loss: {avg_total:.4f}, Seg Loss: {avg_seg:.4f}, Edge Loss: {avg_edge:.4f}, Time: {epoch_time:.2f} sec")

    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")
        print(f"Checkpoint saved at epoch {epoch+1}")

torch.save(model.state_dict(), "unet_256_100ep_2headed_2025-XX-XX.pth")  # update the date
print("Model saved")

total_time = time.time() - total_start_time
print(f"\nTotal training time: {total_time:.2f} seconds")
print(f"Total training time: {total_time/60:.2f} minutes")