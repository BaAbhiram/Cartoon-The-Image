import os
import torch
import cv2
import numpy as np
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
from network.Transformer import Transformer
from torchvision import transforms
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
def load_model(model_path):
    model = Transformer()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    return model

# Image preprocessing
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)

# Save output image
def save_output_image(tensor, output_path):
    tensor = tensor.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
    tensor = (tensor * 0.5 + 0.5) * 255  # Denormalize
    tensor = np.clip(tensor, 0, 255).astype(np.uint8)
    Image.fromarray(tensor).save(output_path)

# Cartoonify function
def cartoonify(image_path, model_path, output_path):
    model = load_model(model_path)
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image_tensor)
    save_output_image(output, output_path)
    return output_path

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded", 400
        file = request.files["file"]
        if file.filename == "":
            return "No file selected", 400
        
        filename = secure_filename(file.filename)
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        output_filename = "cartoonified_" + filename
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        file.save(input_path)
        
        model_path = "pretrained_models/Hayao_net_G_float.pth"  # Change model if needed
        cartoonify(input_path, model_path, output_path)
        
        return render_template("index.html", input_image=input_path, output_image=output_path)
    
    return render_template("index.html")

@app.route("/static/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/static/outputs/<filename>")
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
