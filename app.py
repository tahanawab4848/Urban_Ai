"""
Urban Air Quality Category Prediction - Interactive Streamlit Dashboard.
Learn Depth Academy - Track 1 Capstone (Problem 10).
CPCB / NAQI Standard Multi-Class Environmental Classification.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.preprocess import AirQualityFeatureEngineer, LABEL_TO_INT, INT_TO_LABEL
import __main__
__main__.AirQualityFeatureEngineer = AirQualityFeatureEngineer

from src.utils import CPCB_COLOR_MAP, ORDERED_CATEGORIES

# Page configuration
st.set_page_config(
    page_title="Urban Air Quality Classifier | CPCB Standards",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for CPCB badges and clean typography
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .prediction-label {
        font-size: 1.1rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        font-weight: 600;
    }
    .prediction-value {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0.2rem 0;
    }
    .advisory-card {
        background-color: #F8FAFC;
        border-left: 5px solid #3B82F6;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        margin-top: 1rem;
    }
    .stat-badge {
        background-color: #E2E8F0;
        padding: 0.3rem 0.6rem;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# Cache artifact loading for instant execution
@st.cache_resource
def load_model_and_preprocessor():
    """Load serialized preprocessing pipeline and trained models."""
    prep_path = "artifacts/preprocessor.joblib"
    model_path = "artifacts/model.joblib"

    if not os.path.exists(prep_path) or not os.path.exists(model_path):
        st.error("Model artifacts not found! Please run the training pipeline first.")
        st.stop()

    preprocessor_data = joblib.load(prep_path)
    model_data = joblib.load(model_path)
    return preprocessor_data, model_data


preprocessor_bundle, model_bundle = load_model_and_preprocessor()
preprocessor = preprocessor_bundle["preprocessor"]
engineer = preprocessor_bundle["engineer"]
feature_cols = preprocessor_bundle["feature_columns"]

all_models = model_bundle["all_best_models"]
champion_name = model_bundle["model_name"]

# Health Advisory Lookup according to CPCB NAQI Guidelines
HEALTH_ADVISORIES = {
    "Good": {
        "summary": "Minimal Health Impact. Air quality is satisfactory and poses little to no risk.",
        "general": "Air quality is ideal for all normal outdoor and physical activities.",
        "sensitive": "No special precautions needed. Clean atmospheric conditions.",
        "action": "Enjoy outdoor activities and ventilate indoor spaces freely.",
        "badge_color": "#00E400",
        "text_color": "#053e05"
    },
    "Satisfactory": {
        "summary": "Minor Breathing Discomfort for highly sensitive individuals.",
        "general": "Air quality is generally acceptable. Outdoor exertion is safe for the majority.",
        "sensitive": "Individuals with extreme respiratory hypersensitivity should monitor minor symptoms.",
        "action": "Standard outdoor activities may continue without modification.",
        "badge_color": "#70A800",
        "text_color": "#ffffff"
    },
    "Moderate": {
        "summary": "Breathing Discomfort to people with lungs, asthma, and heart diseases.",
        "general": "General population is unlikely to experience acute symptoms. Sensitive individuals may feel throat irritation.",
        "sensitive": "Children, the elderly, and individuals with asthma/COPD should reduce prolonged heavy exertion outdoors.",
        "action": "Keep windows closed during peak morning traffic hours if living near arterial roadways.",
        "badge_color": "#E6D800",
        "text_color": "#2c2900"
    },
    "Poor": {
        "summary": "Breathing Discomfort to most people on prolonged exposure.",
        "general": "Healthy adults may experience slight cough, throat irritation, and eye redness after sustained outdoor exertion.",
        "sensitive": "People with heart or lung disease should strictly avoid outdoor activity. Carry rescue inhalers.",
        "action": "Wear N95/FFP2 masks during peak hours. Avoid strenuous morning jogging.",
        "badge_color": "#FF7E00",
        "text_color": "#ffffff"
    },
    "Very Poor": {
        "summary": "Respiratory Illness to the people on prolonged exposure. Significant trigger for cardiac events.",
        "general": "Increased risk of respiratory symptoms in the general public. Outdoor physical activity should be minimized.",
        "sensitive": "Elderly, pregnant women, and vulnerable patients should remain indoors with air purifiers.",
        "action": "Avoid all non-essential outdoor travel. Wear rated protective particulate filtration masks.",
        "badge_color": "#FF0000",
        "text_color": "#ffffff"
    },
    "Severe": {
        "summary": "Health Emergency. Affects healthy people and seriously impacts those with existing diseases.",
        "general": "Hazardous toxic conditions. Serious risk of adverse health impacts for the entire urban population.",
        "sensitive": "Strict indoor confinement. Medical emergency standby recommended for vulnerable patients.",
        "action": "Enforce emergency work-from-home, industrial shutdowns, and ban diesel generator sets.",
        "badge_color": "#7E0023",
        "text_color": "#ffffff"
    }
}

# -------------------------------------------------------------
# Sidebar: Model Diagnostics & Presets
# -------------------------------------------------------------
st.sidebar.markdown("### ⚙️ Model Selection & Info")
selected_model_key = st.sidebar.selectbox(
    "Active Inference Classifier:",
    options=list(all_models.keys()),
    index=list(all_models.keys()).index(champion_name),
    format_func=lambda x: f"{x.replace('_', ' ')} {'⭐ (Champion)' if x == champion_name else ''}"
)
active_model = all_models[selected_model_key]

st.sidebar.markdown("---")
st.sidebar.markdown("### 🌆 Load City Scenarios")
preset_choice = st.sidebar.selectbox(
    "Quick Scenario Presets:",
    ["Custom Inputs", "Clean Coastal Day (Bengaluru/Kochi)", "Moderate Urban Mix (Hyderabad)", "Severe Winter Smog (Delhi/Kanpur)", "Industrial High SO2/NO2 Zone"]
)

# Set defaults based on preset
if preset_choice == "Clean Coastal Day (Bengaluru/Kochi)":
    d_pm25, d_pm10, d_no2, d_so2, d_co, d_o3, d_nh3 = 18.0, 35.0, 12.0, 6.0, 0.4, 22.0, 8.0
elif preset_choice == "Moderate Urban Mix (Hyderabad)":
    d_pm25, d_pm10, d_no2, d_so2, d_co, d_o3, d_nh3 = 62.0, 115.0, 32.0, 14.0, 1.1, 42.0, 19.0
elif preset_choice == "Severe Winter Smog (Delhi/Kanpur)":
    d_pm25, d_pm10, d_no2, d_so2, d_co, d_o3, d_nh3 = 285.0, 420.0, 95.0, 28.0, 4.2, 58.0, 52.0
elif preset_choice == "Industrial High SO2/NO2 Zone":
    d_pm25, d_pm10, d_no2, d_so2, d_co, d_o3, d_nh3 = 110.0, 190.0, 88.0, 65.0, 2.5, 75.0, 38.0
else:
    d_pm25, d_pm10, d_no2, d_so2, d_co, d_o3, d_nh3 = 55.0, 95.0, 25.0, 12.0, 0.9, 35.0, 15.0

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 CPCB NAQI Categories")
cpcb_ref = pd.DataFrame({
    "Category": ORDERED_CATEGORIES,
    "AQI Range": ["0–50", "51–100", "101–200", "201–300", "301–400", "401–500+"],
    "Dominant Pollutant": ["PM2.5 / PM10", "PM2.5 / PM10", "PM2.5 / NO2", "PM2.5 / CO", "PM2.5 / PM10", "PM2.5 / PM10"]
})
st.sidebar.dataframe(cpcb_ref, hide_index=True)

# -------------------------------------------------------------
# Main Application Layout
# -------------------------------------------------------------
st.markdown('<div class="main-title">🌫️ Urban Air Quality Category Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Foundational Machine Learning for CPCB / NAQI Multi-Class Environmental Decision Support</div>', unsafe_allow_html=True)

st.markdown("### 1. Ambient Pollutant Concentrations")
st.caption("Enter observed ground monitoring station parameters (values validated against physical CPCB instruments):")

col1, col2 = st.columns(2)

with col1:
    val_pm25 = st.number_input("PM2.5 - Fine Particles (µg/m³)", min_value=0.0, max_value=1000.0, value=float(d_pm25), step=1.0, help="24-hour mean concentration of particulate matter < 2.5 micrometers.")
    val_pm10 = st.number_input("PM10 - Coarse Particles (µg/m³)", min_value=0.0, max_value=1500.0, value=float(d_pm10), step=1.0, help="24-hour mean concentration of particulate matter < 10 micrometers.")
    val_no2 = st.number_input("NO2 - Nitrogen Dioxide (µg/m³)", min_value=0.0, max_value=600.0, value=float(d_no2), step=1.0, help="24-hour mean concentration of Nitrogen Dioxide.")
    val_so2 = st.number_input("SO2 - Sulfur Dioxide (µg/m³)", min_value=0.0, max_value=500.0, value=float(d_so2), step=1.0, help="24-hour mean concentration of Sulfur Dioxide.")

with col2:
    val_co = st.number_input("CO - Carbon Monoxide (mg/m³)", min_value=0.0, max_value=50.0, value=float(d_co), step=0.1, help="8-hour mean concentration of Carbon Monoxide.")
    val_o3 = st.number_input("O3 - Ground-Level Ozone (µg/m³)", min_value=0.0, max_value=500.0, value=float(d_o3), step=1.0, help="8-hour mean concentration of Ozone.")
    val_nh3 = st.number_input("NH3 - Ammonia (µg/m³)", min_value=0.0, max_value=600.0, value=float(d_nh3), step=1.0, help="24-hour mean concentration of Ammonia.")
    val_month = st.slider("Monitoring Month", min_value=1, max_value=12, value=11, help="Month index (affects meteorological seasonality flags).")

# -------------------------------------------------------------
# Input Validation & Aerosol Ratio Verification
# -------------------------------------------------------------
validation_passed = True
if val_pm25 > val_pm10 and val_pm10 > 0:
    st.warning("⚠️ **Atmospheric Consistency Notice:** PM2.5 concentration exceeds PM10. Because PM2.5 is physically a subset of PM10, verify if sensor calibration anomaly exists.")
if any(v < 0 for v in [val_pm25, val_pm10, val_no2, val_so2, val_co, val_o3, val_nh3]):
    st.error("❌ **Invalid Input:** Pollutant concentrations cannot be negative.")
    validation_passed = False

if validation_passed:
    # Construct input dataframe
    input_data = pd.DataFrame([{
        "PM2.5": val_pm25,
        "PM10": val_pm10,
        "NO2": val_no2,
        "SO2": val_so2,
        "CO": val_co,
        "O3": val_o3,
        "NH3": val_nh3,
        "Month": val_month,
        "DayOfWeek": 2
    }])

    # Apply feature engineering transformer
    engineered_input = engineer.transform(input_data)
    # Apply standard scaling preprocessor
    scaled_input = preprocessor.transform(engineered_input)

    # Perform inference
    predicted_int = int(active_model.predict(scaled_input)[0])
    predicted_category = INT_TO_LABEL[predicted_int]

    # Predict class probabilities if model supports it
    if hasattr(active_model, "predict_proba"):
        probabilities = active_model.predict_proba(scaled_input)[0]
    else:
        # Fallback uniform/one-hot for non-probabilistic models
        probabilities = np.zeros(len(ORDERED_CATEGORIES))
        probabilities[predicted_int] = 1.0

    # Display prediction results
    st.markdown("---")
    st.markdown("### 2. Classification Results & CPCB Severity Assessment")

    res_col1, res_col2 = st.columns([1, 1.2])

    with res_col1:
        advisory_info = HEALTH_ADVISORIES[predicted_category]
        bg_col = advisory_info["badge_color"]
        tx_col = advisory_info["text_color"]

        st.markdown(f"""
        <div class="prediction-box" style="background-color: {bg_col}; color: {tx_col};">
            <div class="prediction-label">CPCB NAQI Predicted Air Quality Category</div>
            <div class="prediction-value">{predicted_category}</div>
            <div style="font-size: 0.95rem; font-weight: 600;">{advisory_info['summary']}</div>
        </div>
        """, unsafe_allow_html=True)

        # Derived indicators
        pm_ratio = val_pm25 / max(val_pm10, 1e-4)
        ratio_desc = "Fine Aerosols Dominant (Combustion / Vehicular)" if pm_ratio > 0.6 else "Coarse Dust Dominant (Soil / Windblown)"
        st.markdown(f"**PM2.5 / PM10 Aerosol Ratio:** `{pm_ratio:.2f}` — *{ratio_desc}*")

    with res_col2:
        st.markdown("**Category Probability Distribution:**")
        prob_df = pd.DataFrame({
            "Category": ORDERED_CATEGORIES,
            "Probability": probabilities
        })

        for cat, prob in zip(ORDERED_CATEGORIES, probabilities):
            pct = prob * 100
            bar_color = CPCB_COLOR_MAP[cat]
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; font-size: 0.9rem; margin-bottom: 2px;">
                <span style="font-weight: 600;">{cat}</span>
                <span>{pct:.1f}%</span>
            </div>
            <div style="background-color: #E2E8F0; border-radius: 4px; height: 10px; width: 100%; margin-bottom: 8px;">
                <div style="background-color: {bar_color}; width: {pct}%; height: 10px; border-radius: 4px;"></div>
            </div>
            """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # Clinical Health Advisories
    # -------------------------------------------------------------
    st.markdown("### 3. Clinical & Environmental Health Advisory")
    adv_col1, adv_col2, adv_col3 = st.columns(3)

    with adv_col1:
        st.markdown("#### 🏃 General Public")
        st.info(advisory_info["general"])

    with adv_col2:
        st.markdown("#### 🫁 Sensitive Populations")
        st.warning(advisory_info["sensitive"])

    with adv_col3:
        st.markdown("#### 🛡️ Protective Measures")
        st.success(advisory_info["action"])

    # -------------------------------------------------------------
    # "What-If" Sensitivity Simulator
    # -------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 4. Interactive 'What-If' Emission Reduction Simulator")
    st.caption("Simulate policy interventions (e.g. traffic odd-even rules, construction dust bans, industrial curtailment):")

    reduction_pct = st.slider("Hypothetical Particulate Reduction (%):", min_value=0, max_value=80, value=25, step=5)
    if reduction_pct > 0:
        factor = (100 - reduction_pct) / 100.0
        sim_input_data = pd.DataFrame([{
            "PM2.5": val_pm25 * factor,
            "PM10": val_pm10 * factor,
            "NO2": val_no2 * factor,
            "SO2": val_so2 * factor,
            "CO": val_co * factor,
            "O3": val_o3,
            "NH3": val_nh3,
            "Month": val_month,
            "DayOfWeek": 2
        }])
        sim_engineered = engineer.transform(sim_input_data)
        sim_scaled = preprocessor.transform(sim_engineered)
        sim_pred_int = int(active_model.predict(sim_scaled)[0])
        sim_pred_cat = INT_TO_LABEL[sim_pred_int]

        st.success(f"🌱 **Policy Impact:** Reducing combustion emissions by **{reduction_pct}%** shifts predicted air quality from **{predicted_category}** to **{sim_pred_cat}**!")
