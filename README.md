# House Price Predictor 🇮🇳

A beginner-friendly ML project that predicts Indian house prices using
**Linear Regression** and **Random Forest**.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (saves house_price_model.pkl)
python train.py

# 3. Single prediction from CLI
python predict.py

# 4. Streamlit web app
streamlit run app.py
```

## Project Structure

```
house-price-predictor/
├── data/
│   └── houses.csv          # 545 houses (from Kaggle)
├── train.py                # Train LR + RF, compare, save best
├── predict.py              # CLI single prediction
├── app.py                  # Streamlit web interface
├── requirements.txt
└── house_price_model.pkl   # Saved after running train.py
```

## Models Trained

| Model            | MAE     | RMSE    | R²     |
|------------------|---------|---------|--------|
| Linear Regression| ~₹13L   | ~₹18L   | ~0.65  |
| Random Forest    | ~₹7L    | ~₹11L   | ~0.87  |

> The best model is automatically saved as `house_price_model.pkl`.

## Features Used

- **Numerical**: area, bedrooms, bathrooms, stories, parking
- **Categorical**: mainroad, guestroom, basement, hotwaterheating,
  airconditioning, prefarea, furnishingstatus

## Tech Stack

Python · Pandas · NumPy · scikit-learn · Streamlit
