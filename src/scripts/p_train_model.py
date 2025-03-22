#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import os
import logging
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json

# Configuration du logging
logging.basicConfig(
    filename="logs/train.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_data():
    """Charge les données splittées."""
    files = {
        "X_train": "data/processed/X_train.csv",
        "X_test": "data/processed/X_test.csv",
        "y_train": "data/processed/y_train.csv",
        "y_test": "data/processed/y_test.csv"
    }
    
    for name, path in files.items():
        if not os.path.exists(path):
            logging.error(f"Fichier manquant : {path}")
            raise FileNotFoundError(f"Fichier manquant : {path}")
    
    logging.info("Chargement des données splittées")
    X_train = pd.read_csv(files["X_train"], low_memory=False)
    X_test = pd.read_csv(files["X_test"], low_memory=False)
    y_train = pd.read_csv(files["y_train"])
    y_test = pd.read_csv(files["y_test"])
    
    logging.info(f"X_train : {X_train.shape}, X_test : {X_test.shape}, y_train : {y_train.shape}, y_test : {y_test.shape}")
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Entraîne un modèle RandomForestRegressor."""
    logging.info("Début de l'entraînement du modèle Random Forest")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train.values.ravel())
    logging.info("Entraînement terminé")
    return rf

def evaluate_model(model, X_test, y_test):
    """Évalue le modèle et calcule les métriques."""
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    logging.info(f"Évaluation : R² = {r2:.4f}, MSE = {mse:.4f}")
    return y_pred, {"r2": r2, "mse": mse}

def save_outputs(model, y_pred, metrics):
    """Sauvegarde le modèle, les métriques et les prédictions."""
    output_dirs = {
        "models": Path("models"),
        "metrics": Path("metrics"),
        "data": Path("data")
    }
    for dir_path in output_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarde du modèle
    model_file = output_dirs["models"] / "rf_model.pkl"
    joblib.dump(model, model_file)
    logging.info(f"Modèle sauvegardé dans {model_file}")
    
    # Sauvegarde des métriques
    metrics_file = output_dirs["metrics"] / "scores.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
    logging.info(f"Métriques sauvegardées dans {metrics_file}")
    
    # Sauvegarde des prédictions
    predictions_file = output_dirs["data"] / "predictions.csv"
    predictions = pd.DataFrame({"y_true": y_test.values.ravel(), "y_pred": y_pred})
    predictions.to_csv(predictions_file, index=False)
    logging.info(f"Prédictions sauvegardées dans {predictions_file}")

def main():
    logging.info("Début de l'entraînement et évaluation du modèle")
    
    # Charger les données
    X_train, X_test, y_train, y_test = load_data()
    
    # Entraîner le modèle
    rf = train_model(X_train, y_train)
    
    # Évaluer le modèle
    y_pred, metrics = evaluate_model(rf, X_test, y_test)
    
    # Sauvegarder les résultats
    save_outputs(rf, y_pred, metrics)
    
    logging.info("Entraînement et évaluation terminés avec succès")

if __name__ == "__main__":
    main()