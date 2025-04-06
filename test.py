import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pandas as pd

# Load disease info
csv_path = "corrected_disease.csv"
disease_info = pd.read_csv(csv_path)

# Load model
model_path = "plant_disease_model.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(pretrained=False)
num_classes = len(disease_info)
model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Test Image (Change to actual image path)
image_path = "/home/mhdyahiya/Downloads/Leaf/4/IMG20241218111336.jpg"
image = Image.open(image_path).convert("RGB")
input_data = transform(image).unsqueeze(0).to(device)

# Run Prediction
with torch.no_grad():
    output = model(input_data)
pred = torch.argmax(output).item()

# Fetch corresponding plant name and disease
disease_row = disease_info.iloc[pred]
print(f"Predicted Plant: {disease_row['plant_name']}")
print(f"Predicted Disease: {disease_row['disease_name']}")
print(f"Description: {disease_row['description']}")
print(f"Precautions: {disease_row['precaution']}")
