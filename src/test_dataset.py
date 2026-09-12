from coastline_dataset import CoastlineDataset

dataset = CoastlineDataset(
    image_dir="dataset/images",
    mask_dir="dataset/masks",
    img_size=(256, 256)
)

print("Number of pairs:", len(dataset))

image, mask = dataset[0]

print("Image shape:", image.shape)
print("Mask shape:", mask.shape)
print("Image min/max:", image.min().item(), image.max().item())
print("Mask min/max:", mask.min().item(), mask.max().item())