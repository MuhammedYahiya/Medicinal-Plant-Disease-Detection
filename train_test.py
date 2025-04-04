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
        print(f"📂 Processing {plant_name} - Files: {os.listdir(plant_path)}")

        # Separate healthy and diseased images
        healthy_images = [img for img in os.listdir(plant_path) if "healthy" in img.lower()]
        diseased_images = [img for img in os.listdir(plant_path) if "diseased" in img.lower() or "disease" in img.lower()]

        # Check if folders contain images
        if not healthy_images and not diseased_images:
            print(f"⚠️ Warning: No valid images found in {plant_name}")
            continue

        random.shuffle(healthy_images)
        random.shuffle(diseased_images)

        # Function to split images
        def split_images(image_list):
            split_index = int(len(image_list) * train_ratio)
            return image_list[:split_index], image_list[split_index:]

        train_healthy, test_healthy = split_images(healthy_images)
        train_diseased, test_diseased = split_images(diseased_images)

        # Merge both groups
        train_images = train_healthy + train_diseased
        test_images = test_healthy + test_diseased

        # Create plant subfolders in train & test
        os.makedirs(os.path.join(train_folder, plant_name), exist_ok=True)
        os.makedirs(os.path.join(test_folder, plant_name), exist_ok=True)

        # Copy images instead of moving
        for img in train_images:
            shutil.copy2(os.path.join(plant_path, img), os.path.join(train_folder, plant_name, img))

        for img in test_images:
            shutil.copy2(os.path.join(plant_path, img), os.path.join(test_folder, plant_name, img))

print("✅ Dataset split into train & test successfully (Balanced Healthy & Diseased)!")
