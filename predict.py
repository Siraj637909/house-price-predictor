"""
Predict house price from user input.
Uses the model saved by train.py  (house_price_model.pkl)
"""
import pandas as pd
import joblib

model = joblib.load("house_price_model.pkl")

house = pd.DataFrame([{
    "area":            1500,
    "bedrooms":        3,
    "bathrooms":       2,
    "stories":         2,
    "mainroad":        "yes",
    "guestroom":       "no",
    "basement":        "no",
    "hotwaterheating": "no",
    "airconditioning": "yes",
    "parking":         1,
    "prefarea":        "yes",
    "furnishingstatus": "semi-furnished",
}])

prediction = model.predict(house)
print(f"Predicted house price: ₹{prediction[0]:,.0f}")
