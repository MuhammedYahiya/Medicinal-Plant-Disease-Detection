import os
import pandas as pd
from torchvision import datasets

# ✅ Paths
train_dir = "dataset/train"  # Change if needed
csv_path = "corrected_disease.csv"
output_csv_path = "corrected_disease.csv"  # Output file

# ✅ Load dataset using ImageFolder
train_dataset = datasets.ImageFolder(root=train_dir)
class_to_idx = train_dataset.class_to_idx

# ✅ Load and align CSV
disease_info = pd.read_csv(csv_path)

# Check if all folder names in CSV exist in dataset class mapping
csv_folder_names = set(disease_info["folder_name"])
missing_folders = csv_folder_names - set(class_to_idx.keys())
if missing_folders:
    print(f"⚠️ Warning: These folder names are in CSV but not in dataset: {missing_folders}")

# Add index for sorting based on model's class index
disease_info["index"] = disease_info["folder_name"].map(class_to_idx)

# Drop rows with unmatched folder names to avoid misalignment
disease_info = disease_info.dropna(subset=["index"])

# ✅ Sort and save
disease_info = disease_info.sort_values("index").reset_index(drop=True)
disease_info.drop(columns=["index"], inplace=True)
disease_info.to_csv(output_csv_path, index=False)

print(f"✅ CSV has been aligned and saved to: {output_csv_path}")
print("✅ Final class order preview:")
for i, row in disease_info.iterrows():
    print(f"Class {i} → {row['folder_name']} ({row['plant_name']} - {row['disease_name']})")
