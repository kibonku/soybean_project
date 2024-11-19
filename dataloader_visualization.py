import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json

class SoybeanDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        # Load annotations
        with open(annotation_file) as f:
            self.annotations = json.load(f)

        # Create a list of image files and corresponding annotations
        self.images = []
        self.boxes = []

        for item in self.annotations['images']:
            file_name = item['file_name']
            image_id = item['id']
            self.images.append(file_name)

            # Find corresponding annotations
            image_boxes = []
            for annotation in self.annotations['annotations']:
                if annotation['image_id'] == image_id:
                    bbox = annotation['bbox']  # COCO format: [xmin, ymin, width, height]
                    image_boxes.append(bbox)
            self.boxes.append(image_boxes)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        image_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(image_path).convert("RGB")

        # Get bounding boxes
        boxes = torch.as_tensor(self.boxes[idx], dtype=torch.float32)

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, boxes

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

#* DadtaLoader Implementation
# Paths to your image directory and annotation file
image_dir = './sample_images'
annotation_file = './annotations/_annotations.coco.json'


# Create the dataset and DataLoader
dataset = SoybeanDataset(image_dir, annotation_file, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

# Test the DataLoader
for images, boxes in dataloader:
    print(f"Batch of images shape: {images.shape}")
    print(f"Batch of bounding boxes: {boxes}")
    break

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

def visualize_sample(image, boxes):
    # Convert the image from tensor format to a format suitable for Matplotlib
    image = image.permute(1, 2, 0).numpy()

    # Create a Matplotlib figure
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Plot each bounding box
    for box in boxes:
        xmin, ymin, width, height = box
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()

# Randomly select 5 samples and visualize them
for _ in range(5):
    idx = random.randint(0, len(dataset) - 1)
    image, boxes = dataset[idx]
    visualize_sample(image, boxes)

