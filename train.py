"""
House Price Predictor — Linear Regression & Random Forest
Dataset: 545 houses with area, bedrooms, bathrooms, etc.
"""
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ── 1. Load ──────────────────────────────────────────────
df = pd.read_csv("data/houses.csv")

print("=" * 50)
print("HOUSE PRICE PREDICTOR")
print("=" * 50)
print(f"\nDataset shape : {df.shape}")
print(f"Missing values:\n{df.isnull().sum()}")
print(f"\nFirst 3 rows:")
print(df.head(3))


# ── 2. X and y ──────────────────────────────────────────
X = df.drop("price", axis=1)
y = df["price"]

print(f"\nFeatures: {list(X.columns)}")
print(f"Target  : price")


# ── 3. Train / test split ────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")


# ── 4. Feature groups ────────────────────────────────────
numerical_features = [
    "area", "bedrooms", "bathrooms", "stories", "parking"
]

categorical_features = [
    "mainroad", "guestroom", "basement",
    "hotwaterheating", "airconditioning", "prefarea",
    "furnishingstatus"
]

print(f"\nNumerical features  ({len(numerical_features)}): {numerical_features}")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")


# ── 5. Numerical pipeline ────────────────────────────────
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])


# ── 6. Categorical pipeline ──────────────────────────────
categorical_pipeline = Pipeline([
    ("imputer",  SimpleImputer(strategy="most_frequent")),
    ("onehot",   OneHotEncoder(handle_unknown="ignore")),
])


# ── 7. Column transformer ────────────────────────────────
preprocessor = ColumnTransformer([
    ("numeric",     numeric_pipeline,     numerical_features),
    ("categorical", categorical_pipeline, categorical_features),
])


# ── 8. Linear Regression ─────────────────────────────────
lr_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression()),
])

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

lr_mae  = mean_absolute_error(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2   = r2_score(y_test, lr_pred)


# ── 9. Random Forest ─────────────────────────────────────
rf_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=200, random_state=42)),
])

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_mae  = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2   = r2_score(y_test, rf_pred)


# ── 10. Results ──────────────────────────────────────────
print("\n" + "=" * 50)
print("RESULTS")
print("=" * 50)
print(f"\n{'Metric':<10} {'Linear Reg':>12} {'Random Forest':>14}")
print("-" * 38)
print(f"{'MAE':<10} {lr_mae:>12,.0f} {rf_mae:>14,.0f}")
print(f"{'RMSE':<10} {lr_rmse:>12,.0f} {rf_rmse:>14,.0f}")
print(f"{'R²':<10} {lr_r2:>12.4f} {rf_r2:>14.4f}")

print("\nSample predictions (first 5 test houses):")
results = pd.DataFrame({
    "Actual":    y_test.values[:5],
    "LR_Pred":   lr_pred[:5].astype(int),
    "RF_Pred":   rf_pred[:5].astype(int),
})
print(results.to_string(index=False))


# ── 11. Save the best model ──────────────────────────────
if rf_r2 >= lr_r2:
    best = rf_model
    name = "Random Forest"
else:
    best = lr_model
    name = "Linear Regression"

joblib.dump(best, "house_price_model.pkl")
print(f"\nSaved best model ({name}) → house_price_model.pkl")


# ── 12. Feature importance (Random Forest) ───────────────
print("\n" + "=" * 50)
print("RANDOM FOREST — Feature Importance")
print("=" * 50)

# Get feature names after preprocessing
ohe = rf_model.named_steps["preprocessor"] \
               .named_transformers_["categorical"] \
               .named_steps["onehot"]
cat_names = ohe.get_feature_names_out(categorical_features).tolist()
all_names = numerical_features + cat_names

importances = rf_model.named_steps["model"].feature_importances_
fi = pd.Series(importances, index=all_names).sort_values(ascending=False)

print(fi.head(15).to_string())
