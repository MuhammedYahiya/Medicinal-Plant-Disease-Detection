import torch
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn

# ✅ Load Model
# ✅ Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Define model (MUST MATCH TRAINED STRUCTURE)
num_classes = 12  # Update based on dataset
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Sequential(
    nn.Dropout(0.3),  # Same dropout as training
    nn.Linear(model.last_channel, num_classes)
)
model = model.to(device)

# ✅ Load trained model weights
model.load_state_dict(torch.load("best_plant_disease_model.pt", map_location=device))
model.eval()  # Set model to evaluation mode

print("✅ Model loaded successfully!")

# ✅ Load Test Data
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

test_dir = "dataset/test"
test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ✅ Evaluation Metrics
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ✅ Calculate Accuracy
accuracy = accuracy_score(all_labels, all_preds)
print(f"✅ Test Accuracy: {accuracy * 100:.2f}%")

# ✅ Classification Report
print("\n📊 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

# ✅ Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
