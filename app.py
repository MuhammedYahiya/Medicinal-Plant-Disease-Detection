import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from flask import Flask, request, render_template
from PIL import Image
import pandas as pd

# ✅ Load Disease Information
csv_path = "corrected_disease.csv"
disease_info = pd.read_csv(csv_path)

# ✅ Load Trained Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Set number of classes (MUST match the trained model)
num_classes = 54  # ✅ Set this to 27 to match trained model

# ✅ Define model architecture
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Sequential(
    nn.Dropout(0.3),  # Same dropout used during training
    nn.Linear(model.last_channel, num_classes)
)
model = model.to(device)

# ✅ Load trained model weights
model.load_state_dict(torch.load("best_plant_disease_model.pt", map_location=device))
model.eval()  # Set model to evaluation mode

# ✅ Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # RGB image normalization
])

# ✅ Flask app setup
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Prediction function
def predict_disease(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
    
    probabilities = torch.softmax(output, dim=1)
    confidence, index = torch.max(probabilities, 1)
    return index.item(), confidence.item()

# ✅ Routes
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        image = request.files["image"]
        file_path = os.path.join(UPLOAD_FOLDER, image.filename)
        image.save(file_path)

        pred, confidence = predict_disease(file_path)

        try:
            disease_row = disease_info.iloc[pred]
            return render_template(
                "result.html",
                plant=disease_row["plant_name"],
                title=disease_row["disease_name"],
                desc=disease_row["description"],
                prevent=disease_row["precaution"],
                confidence=f"{confidence * 100:.2f}%",
                image_url=file_path
            )
        except IndexError:
            return f"Prediction index {pred} not found in CSV. Make sure your CSV has 27 entries in correct order."

    return render_template("index.html")

# ✅ Run the app
if __name__ == "__main__":
    app.run(debug=True)
