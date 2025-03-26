import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Define transformations for training (with augmentation)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),            # Resize images to 224x224
    transforms.RandomHorizontalFlip(p=0.5),   # Flip images randomly
    transforms.RandomRotation(20),            # Rotate images randomly by ±20 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Adjust brightness & contrast
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Random zoom
    transforms.ToTensor(),                    # Convert image to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize pixel values
])

# Define transformations for test data (only resize & normalize)
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load datasets
train_dataset = ImageFolder(root="dataset/train", transform=train_transforms)
test_dataset = ImageFolder(root="dataset/test", transform=test_transforms)

# Create DataLoaders for batching
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("✅ Data Preprocessing & Augmentation Applied Successfully!")

# Get a batch of training images
dataiter = iter(train_loader)
images, labels = next(dataiter)

# Convert tensor to numpy array for visualization
def imshow(img):
    img = img.numpy().transpose((1, 2, 0))  # Convert (C, H, W) to (H, W, C)
    img = img * 0.5 + 0.5  # Unnormalize to original range
    plt.imshow(img)
    plt.axis('off')

# Display first 5 images with augmentation
plt.figure(figsize=(10,5))
for i in range(5):
    plt.subplot(1,5,i+1)
    imshow(images[i])
plt.show()

