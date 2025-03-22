#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Charger les données pré-traitées
input_file = "data/processed/DF_2021-23_Processed.csv"
data = pd.read_csv(input_file)

# Séparer les features (X) et la cible (y)
X = data.drop(columns=['Ewltp (g/km)', 'Cn'])  # Features : tout sauf la cible et 'Cn'
y = data['Ewltp (g/km)']  # Cible : émissions CO2 (Ewltp)

# Split en train et test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer le dossier processed si nécessaire
os.makedirs("data/processed", exist_ok=True)

# Sauvegarder les fichiers splittés pour une réutilisation future
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("Données splittées et sauvegardées dans data/processed/ : X_train.csv, X_test.csv, y_train.csv, y_test.csv")