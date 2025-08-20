#!/usr/bin/env python3
"""
Flask web application for Traffic Sign Classification.
"""
import io
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from src.data_handler import preprocess_images
from src.model_handler import ModelInference

app = Flask(__name__)
app.secret_key = "traffic_sign_secret"
MODEL_PATH = "lenet.keras"
SIGN_NAMES_PATH = "signnames.csv"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff"}


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            try:
                # Read image in memory
                in_memory_file = io.BytesIO()
                file.save(in_memory_file)
                data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (32, 32))
                img = preprocess_images(np.array([img]))

                # Run inference
                inference = ModelInference(model_path=MODEL_PATH)
                predictions = inference.predict(img)[0]

                # Get results
                sign_names = pd.read_csv(SIGN_NAMES_PATH)
                top5 = []
                for idx in np.argsort(predictions)[::-1][:5]:
                    label = sign_names.iloc[idx]["SignName"]
                    prob = float(predictions[idx])
                    top5.append((label, f"{prob:.2%}"))

                return render_template("predict.html", top5=top5)
            except Exception as e:
                flash(f"An error occurred: {e}")
                return redirect(request.url)

    return render_template("predict.html")


if __name__ == "__main__":
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Model '{MODEL_PATH}' not found. Train the model first."
        )
    app.run(debug=True)
