"""Training and inference helpers for the mobile price classifier."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["ram", "battery", "screen", "cores", "storage", "camera", "resolution", "five_g"]
CATEGORIES = {
    "Budget": ("Under ₹15,000", "Entry-level hardware for everyday essentials."),
    "Mid-range": ("₹15,000–₹29,999", "Balanced features for most people."),
    "Premium": ("₹30,000–₹59,999", "High-end performance and advanced features."),
    "Flagship": ("₹60,000+", "Top-tier hardware and premium capabilities."),
}

def build_classifier(random_state: int = 42):
    return make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=240, min_samples_leaf=2, class_weight="balanced", random_state=random_state, n_jobs=-1))

def build_demo_model(random_state: int = 42):
    """Train a reproducible preview model on generated, rule-labeled examples."""
    rng = np.random.default_rng(random_state)
    n = 5000
    x = pd.DataFrame({
        "ram": rng.integers(512, 12001, n), "battery": rng.integers(1800, 7001, n),
        "screen": rng.uniform(4.3, 7.4, n), "cores": rng.choice([2, 4, 6, 8], n),
        "storage": rng.choice([8, 16, 32, 64, 128, 256, 512], n), "camera": rng.integers(2, 201, n),
        "resolution": rng.choice([320, 480, 720, 1080, 1440, 2160], n), "five_g": rng.integers(0, 2, n),
    })
    # A smooth hardware-value score generates plausible illustrative labels.
    score = (x.ram / 12000 * 2.1 + x.storage / 512 * 1.5 + x.camera / 200 * .8 + x.resolution / 2160 * .7 + x.cores / 8 * .7 + x.five_g * .45 + x.battery / 7000 * .35 + x.screen / 7.5 * .3 + rng.normal(0, .28, n))
    y = pd.cut(score, [-np.inf, 1.65, 2.65, 3.65, np.inf], labels=list(CATEGORIES)).astype(str)
    return build_classifier(random_state).fit(x[FEATURES], y)

def predict_one(model, values: dict):
    row = pd.DataFrame([{name: values[name] for name in FEATURES}], columns=FEATURES)
    probabilities = model.predict_proba(row)[0]
    classes = model.classes_
    best = int(np.argmax(probabilities))
    return str(classes[best]), float(probabilities[best]), {str(c): float(p) for c, p in zip(classes, probabilities)}

def train_from_csv(path: str, output: str = "model.joblib"):
    """Train on a labeled CSV containing FEATURES and a price_range column (0–3)."""
    import joblib
    data = pd.read_csv(path)
    missing = set(FEATURES + ["price_range"]) - set(data.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")
    labels = {0: "Budget", 1: "Mid-range", 2: "Premium", 3: "Flagship", "0": "Budget", "1": "Mid-range", "2": "Premium", "3": "Flagship"}
    y = data.price_range.map(labels).fillna(data.price_range.astype(str))
    model = build_classifier().fit(data[FEATURES], y)
    joblib.dump(model, output)
    return model
