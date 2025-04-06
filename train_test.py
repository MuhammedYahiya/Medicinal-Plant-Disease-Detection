import os
import shutil
import random

# Define dataset path
dataset_folder = "SET"
train_folder = os.path.join(dataset_folder, "train")
test_folder = os.path.join(dataset_folder, "test")

# Define train-test split ratio
train_ratio = 0.8  # 80% train, 20% test

# Create train & test folders
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Process each plant folder
for plant_name in os.listdir(dataset_folder):
    plant_path = os.path.join(dataset_folder, plant_name)

    # Skip train/test folders
    if plant_name in ["train", "test"]:
        continue

    if os.path.isdir(plant_path):
        print(f"📂 Processing {plant_name}")

        # Separate healthy and diseased images
        healthy_images = [img for img in os.listdir(plant_path) if "healthy" in img.lower()]
        diseased_images = [img for img in os.listdir(plant_path) if "diseased" in img.lower() or "disease" in img.lower()]

        # Function to extract disease type from filename
        def get_disease_name(filename):
            # Expected format: diseased_diseasename_x.jpg
            parts = filename.lower().split("_")
            if len(parts) >= 2:
                return parts[1]  # disease name
            return "unknown"

        # Organize images by category
        disease_dict = {}

        # Add healthy under specific folder name
        if healthy_images:
            disease_dict[f"{plant_name}_healthy"] = healthy_images

        # Group diseased by disease type
        for img in diseased_images:
            disease_name = get_disease_name(img)
            key = f"{plant_name}_{disease_name}"
            disease_dict.setdefault(key, []).append(img)

        # For each category (healthy or disease type)
        for category, images in disease_dict.items():
            random.shuffle(images)
            split_index = int(len(images) * train_ratio)
            train_imgs = images[:split_index]
            test_imgs = images[split_index:]

            # Create directories
            os.makedirs(os.path.join(train_folder, category), exist_ok=True)
            os.makedirs(os.path.join(test_folder, category), exist_ok=True)

            # Copy images
            for img in train_imgs:
                shutil.copy2(os.path.join(plant_path, img), os.path.join(train_folder, category, img))

            for img in test_imgs:
                shutil.copy2(os.path.join(plant_path, img), os.path.join(test_folder, category, img))

print("✅ Dataset split with healthy/diseased categories separated!")
