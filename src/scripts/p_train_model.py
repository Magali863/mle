#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json

# Charger les données splittées
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv")
y_test = pd.read_csv("data/processed/y_test.csv")

# Entraîner le modèle Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train.values.ravel())  # .values.ravel() pour convertir y_train en 1D

# Faire des prédictions sur le test set
y_pred = rf.predict(X_test)

# Calculer les métriques
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"Score R² : {r2}")
print(f"Mean Squared Error : {mse}")

# Sauvegarder le modèle
output_dir = "models"
os.makedirs(output_dir, exist_ok=True)
model_filename = f"{output_dir}/rf_model.pkl"
joblib.dump(rf, model_filename)
print(f"Modèle sauvegardé : {model_filename}")

# Sauvegarder les métriques dans un fichier JSON
metrics = {"r2": r2, "mse": mse}
os.makedirs("metrics", exist_ok=True)
with open("metrics/scores.json", "w") as f:
    json.dump(metrics, f, indent=4)
print("Métriques sauvegardées dans metrics/scores.json")

# Sauvegarder les prédictions
predictions = pd.DataFrame({"y_true": y_test.values.ravel(), "y_pred": y_pred})
predictions.to_csv("data/predictions.csv", index=False)
print("Prédictions sauvegardées dans data/predictions.csv")