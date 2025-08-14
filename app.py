
import os
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image

# TensorFlow / Keras (ImageNet demo)
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2, preprocess_input, decode_predictions
)

# ---- Config ----
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SECRET_KEY"] = "demo-only-not-secret"  # for flash messages

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---- Load model once on startup ----
model = MobileNetV2(weights="imagenet")

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_image(path: str) -> np.ndarray:
    """Load and preprocess image to 224x224 RGB for MobileNetV2."""
    img = Image.open(path).convert("RGB").resize((224, 224))
    x = np.array(img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        flash("No file part in request.")
        return redirect(url_for("home"))

    file = request.files["image"]
    if file.filename == "":
        flash("No file selected.")
        return redirect(url_for("home"))

    if not allowed_file(file.filename):
        flash("Unsupported file type. Please upload a PNG/JPG/GIF/BMP/WEBP.")
        return redirect(url_for("home"))

    # Save upload
    filename = secure_filename(f"{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}_{file.filename}")
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    # Prepare and predict
    x = prepare_image(save_path)
    preds = model.predict(x)
    top5 = decode_predictions(preds, top=5)[0]  # [(class_id, class_name, score), ...]

    results = [{"id": cid, "label": label.replace("_", " ").title(), "score": float(score)}
               for cid, label, score in top5]

    return render_template("result.html",
                           image_url=url_for("uploaded_file", filename=filename),
                           results=results,
                           filename=filename)

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return app.send_static_file(os.path.join("..", "uploads", filename))  # not used (see below)

# Safer static serve for uploads (avoids above relative approach)
from flask import send_from_directory
@app.route("/file/<path:filename>")
def file_serve(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    # Note: Debug server is for local dev only
    app.run(debug=True)
