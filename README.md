# Mobile Device Price Prediction System

A mobile phone price category estimator for Indian price bands, shown in INR (₹), using a scikit-learn Random Forest classifier. It includes a Streamlit app and a standalone responsive Flask website.

## Run the app

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Run the website

```bash
python -m pip install -r requirements.txt
python web_app.py
```

Then open <http://127.0.0.1:5000>. The website is in `website/`; Flask serves the page and the prediction API.

The app runs immediately with a deterministic **demo model** trained on generated examples. Its predictions are illustrative, not market-validated. For deployment, train on a representative dataset of device specifications and known price tiers.

## Train with labeled data

Prepare a CSV with these columns:

`ram,battery,screen,cores,storage,camera,resolution,five_g,price_range`

Feature units: RAM in MB, battery in mAh, screen in inches, storage in GB, camera in MP, display resolution as vertical pixels, and `five_g` as 0/1. `price_range` may be 0–3 labels (mapped to Budget, Mid-range, Premium, Flagship) or those category names. Displayed bands are Indian-market guide ranges, not a live listing price.

```bash
python train_model.py your_devices.csv --output model.joblib
streamlit run app.py
```

The app automatically uses `model.joblib` when present. Keep training and inference data representative of the same device market and units.
