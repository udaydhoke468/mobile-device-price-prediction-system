"""Serve the standalone website and its mobile price prediction API."""
from pathlib import Path

import joblib
from flask import Flask, jsonify, request, send_from_directory

from model import CATEGORIES, build_demo_model, predict_one

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.joblib"
app = Flask(__name__, static_folder="website", static_url_path="")
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else build_demo_model()

@app.get("/")
def home():
    return send_from_directory(app.static_folder, "index.html")

@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    try:
        values = {
            "ram": int(payload["ram"]), "battery": int(payload["battery"]),
            "screen": float(payload["screen"]), "cores": int(payload["cores"]),
            "storage": int(payload["storage"]), "camera": int(payload["camera"]),
            "resolution": int(payload["resolution"]), "five_g": int(bool(payload["five_g"])),
        }
        category, confidence, probabilities = predict_one(model, values)
        band, description = CATEGORIES[category]
        return jsonify(category=category, band=band, description=description,
                       confidence=confidence, probabilities=probabilities)
    except (KeyError, TypeError, ValueError) as error:
        return jsonify(error=f"Invalid device specifications: {error}"), 400

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
