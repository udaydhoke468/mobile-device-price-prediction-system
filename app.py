"""Mobile price category predictor — Streamlit demo and inference UI."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from model import CATEGORIES, FEATURES, build_demo_model, predict_one

MODEL_PATH = Path(__file__).with_name("model.joblib")

st.set_page_config(page_title="Mobile Price Predictor", page_icon="📱", layout="wide")

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH), False
    return build_demo_model(), True

model, is_demo = load_model()

st.markdown("<div class='eyebrow'>DEVICE VALUE · ML ESTIMATOR</div>", unsafe_allow_html=True)
st.title("What’s your phone worth?")
st.write("Enter a few hardware details to estimate its market price tier.")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {font-family:Manrope,sans-serif}
.eyebrow {font:500 11px 'DM Mono',monospace;letter-spacing:2px;color:#5e6dff;margin-top:12px}
[data-testid="stAppViewContainer"] {background:radial-gradient(ellipse at 85% 2%,#e9eaff 0,transparent 30%),#f7f8fc}
.result {padding:25px 28px;background:#fff;border:1px solid #e8eaf2;border-radius:20px;box-shadow:0 12px 40px #2329560b}
.tier {font-size:32px;font-weight:800;letter-spacing:-1px;margin:4px 0;color:#21243b}
.muted {color:#727790;font-size:14px}
div[data-testid="stForm"] {background:#fff;padding:22px 24px;border:1px solid #e8eaf2;border-radius:20px}
</style>""", unsafe_allow_html=True)

if is_demo:
    st.info("Demo model active · trained on generated examples for a working preview. Upload a real labeled dataset and train a production model from the project scripts.", icon="ℹ️")

left, right = st.columns([1.1, .9], gap="large")
with left:
    st.subheader("Device specifications")
    with st.form("specs"):
        a, b = st.columns(2)
        with a:
            ram = st.slider("Memory (RAM)", 512, 12000, 4000, step=512, format="%d MB")
            battery = st.slider("Battery capacity", 1500, 7000, 4500, step=100, format="%d mAh")
            screen = st.slider("Screen size", 4.0, 7.5, 6.2, step=.1, format="%.1f in")
            cores = st.select_slider("Processor cores", options=[2, 4, 6, 8], value=8)
        with b:
            storage = st.select_slider("Storage", options=[8, 16, 32, 64, 128, 256, 512], value=128, format_func=lambda x: f"{x} GB")
            camera = st.slider("Main camera", 2, 200, 48, step=2, format="%d MP")
            resolution = st.select_slider("Screen resolution", options=[320, 480, 720, 1080, 1440, 2160], value=1080, format_func=lambda x: f"{x}p")
            five_g = st.toggle("5G connectivity", value=True)
        submitted = st.form_submit_button("Estimate price tier", type="primary", use_container_width=True)

with right:
    st.subheader("Estimated category")
    if submitted or "prediction" not in st.session_state:
        values = {"ram": ram, "battery": battery, "screen": screen, "cores": cores, "storage": storage, "camera": camera, "resolution": resolution, "five_g": int(five_g)}
        category, confidence, probabilities = predict_one(model, values)
        st.session_state.update(prediction=category, confidence=confidence, probabilities=probabilities)
    category = st.session_state.prediction
    confidence = st.session_state.confidence
    probabilities = st.session_state.probabilities
    band, description = CATEGORIES[category]
    st.markdown(f"<div class='result'><div class='muted'>PREDICTED PRICE RANGE</div><div class='tier'>{category}</div><div style='font-size:20px;font-weight:700;color:#5e6dff'>{band}</div><p class='muted'>{description}</p><hr style='border:0;border-top:1px solid #eeeef4'><div class='muted'>Model confidence · {confidence:.0%}</div></div>", unsafe_allow_html=True)
    st.write("")
    st.caption("Category probability")
    st.bar_chart(pd.DataFrame({"Probability": probabilities}, index=list(CATEGORIES)), horizontal=True, color="#626cff")
    st.caption("Predictions are indicative and reflect hardware specifications only. Brand, condition, region, and resale demand can change actual prices.")

with st.expander("How this model works"):
    st.write("A Random Forest classifier uses RAM, battery capacity, screen size, processor cores, storage, camera resolution, display resolution, and 5G support. Price tiers are shown in INR. The demo model is trained on generated sample data. For real-world use, train it on representative Indian device listings with known price categories.")
