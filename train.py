import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

FEATURES = ["area", "bedrooms", "bathrooms", "location_score", "garage", "year_built", "floors"]

df = pd.read_csv("dataset.csv")
X = df[FEATURES]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

candidates = {
    "RandomForest":       RandomForestRegressor(n_estimators=200, random_state=42),
    "GradientBoosting":   GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42),
}

best_name, best_pipeline, best_r2 = None, None, -1

print(f"{'Model':<22} {'MAE':>12} {'R²':>8}")
print("-" * 44)

for name, regressor in candidates.items():
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", regressor)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    print(f"{name:<22} ${mae:>10,.0f} {r2:>8.4f}")
    if r2 > best_r2:
        best_r2, best_name, best_pipeline = r2, name, pipeline

print(f"\nBest model: {best_name} (R2={best_r2:.4f})")

with open("model.pkl", "wb") as f:
    pickle.dump({"pipeline": best_pipeline, "features": FEATURES, "model_name": best_name}, f)

print("Model saved as model.pkl")
