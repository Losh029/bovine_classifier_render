import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import base64
from io import BytesIO
from huggingface_hub import hf_hub_download
import os

# CSV path
CSV_PATH = "data/decription.csv"  # forward slashes

# Load CSV data
def load_breed_data():
    return pd.read_csv(CSV_PATH)

# Load model directly from Hugging Face
def load_model(repo_id="Lokeshsasi029/bovine_resnet50", model_filename="bovine_resnet50.pth"):
    # Download model from HF
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if token is None:
        raise ValueError("Set HUGGINGFACE_HUB_TOKEN environment variable before running!")

    model_path = hf_hub_download(repo_id=repo_id, filename=model_filename, token=token)

    # Load PyTorch model
    model = models.resnet50(weights=None)  # replace with your architecture
    model.fc = torch.nn.Linear(model.fc.in_features, 41)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

# Image preprocessing
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return transform(image).unsqueeze(0)

# Decode base64 camera image
def decode_base64_image(data):
    header, encoded = data.split(",", 1)
    data_bytes = base64.b64decode(encoded)
    return Image.open(BytesIO(data_bytes)).convert("RGB")

# Predict function
def predict(image_path_or_pil, model, breed_data, lang="en"):
    if isinstance(image_path_or_pil, str):
        tensor = preprocess_image(image_path_or_pil)
    else:
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        tensor = transform(image_path_or_pil).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, class_idx = torch.max(probs,1)
        confidence = confidence.item()
        class_idx = class_idx.item()

    row = breed_data.iloc[class_idx]
    name_col = f"Name_{lang}"
    desc_col = f"Description_{lang}"
    return {
        "name": row[name_col],
        "description": row[desc_col],
        "confidence": round(confidence*100,2)
    }
