#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import os
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split

# Configuration du logging
logging.basicConfig(
    filename="logs/split.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def split_data(input_file):
    """Sépare les données en ensembles d'entraînement et de test."""
    if not os.path.exists(input_file):
        logging.error(f"Fichier d'entrée manquant : {input_file}")
        raise FileNotFoundError(f"Fichier d'entrée manquant : {input_file}")
    
    logging.info(f"Chargement des données depuis {input_file}")
    data = pd.read_csv(input_file, low_memory=False)
    logging.info(f"Données chargées : {len(data)} lignes, {len(data.columns)} colonnes")
    
    # Séparer les features (X) et la cible (y)
    X = data.drop(columns=['Ewltp (g/km)', 'Cn'])  # Features : tout sauf la cible et 'Cn'
    y = data['Ewltp (g/km)']  # Cible : émissions CO2 (Ewltp)
    logging.info(f"Features sélectionnées : {X.columns.tolist()}")
    logging.info(f"Cible sélectionnée : 'Ewltp (g/km)'")

    # Split en train et test (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info(f"Split effectué : {len(X_train)} lignes pour l'entraînement, {len(X_test)} lignes pour le test")

    return X_train, X_test, y_train, y_test

def save_data(X_train, X_test, y_train, y_test, output_dir="data/processed"):
    """Sauvegarde les ensembles splittés dans des fichiers CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files = {
        "X_train": output_dir / "X_train.csv",
        "X_test": output_dir / "X_test.csv",
        "y_train": output_dir / "y_train.csv",
        "y_test": output_dir / "y_test.csv"
    }
    
    X_train.to_csv(files["X_train"], index=False)
    X_test.to_csv(files["X_test"], index=False)
    y_train.to_csv(files["y_train"], index=False)
    y_test.to_csv(files["y_test"], index=False)
    
    for name, path in files.items():
        logging.info(f"{name} sauvegardé dans {path}")

def main():
    input_file = "data/processed/DF_2021-23_Processed.csv"
    logging.info("Début de la séparation des données")
    
    X_train, X_test, y_train, y_test = split_data(input_file)
    save_data(X_train, X_test, y_train, y_test)
    
    logging.info("Séparation des données terminée avec succès")

if __name__ == "__main__":
    main()