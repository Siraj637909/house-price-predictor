"""
Streamlit web app — enter house details and get a price prediction.
Run with:  streamlit run app.py
"""
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

st.title("🏠 Indian House Price Predictor")
st.markdown("Enter house details below to get an estimated price.")

# ── Model ────────────────────────────────────────────────
import os

MODEL_PATH = "house_price_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.info("Training model for the first time…")
    from train import *
    st.rerun()

model = joblib.load(MODEL_PATH)

# ── Input form ──────────────────────────────────────────
with st.form("house_form"):
    area = st.number_input("Area (sq ft)", 300, 20000, 1500)

    col1, col2, col3 = st.columns(3)
    with col1:
        bedrooms  = st.slider("Bedrooms",  1, 6, 3)
        bathrooms = st.slider("Bathrooms", 1, 4, 2)
    with col2:
        stories   = st.slider("Stories",   1, 4, 2)
        parking   = st.slider("Parking spots", 0, 3, 1)
    with col3:
        furnishing = st.selectbox("Furnishing", [
            "furnished", "semi-furnished", "unfurnished"
        ])

    col4, col5 = st.columns(2)
    with col4:
        mainroad        = st.checkbox("Main Road",        True)
        guestroom       = st.checkbox("Guest Room",       False)
        basement        = st.checkbox("Basement",         False)
    with col5:
        hotwaterheating = st.checkbox("Hot Water Heating", False)
        airconditioning = st.checkbox("Air Conditioning",  True)
        prefarea        = st.checkbox("Preferred Area",    True)

    submitted = st.form_submit_button("💰 Predict Price")

# ── Predict ──────────────────────────────────────────────
if submitted:
    house = pd.DataFrame([{
        "area":             area,
        "bedrooms":         bedrooms,
        "bathrooms":        bathrooms,
        "stories":          stories,
        "mainroad":         "yes" if mainroad        else "no",
        "guestroom":        "yes" if guestroom        else "no",
        "basement":         "yes" if basement          else "no",
        "hotwaterheating":  "yes" if hotwaterheating  else "no",
        "airconditioning":  "yes" if airconditioning  else "no",
        "parking":          parking,
        "prefarea":         "yes" if prefarea         else "no",
        "furnishingstatus": furnishing,
    }])

    prediction = model.predict(house)

    st.success(f"### ₹{prediction[0]:,.0f}")

    st.markdown("---")
    st.markdown("**Input Summary:**")
    st.write(house.iloc[0].to_dict())
