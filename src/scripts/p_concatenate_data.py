#!/usr/bin/env python
# coding: utf-8
"""
Concaténation des datasets bruts pour les années 2021, 2022 et 2023.
Les fichiers d'entrée sont explicitement définis comme DF_2021_Raw.csv, DF_2022_Raw.csv et DF_2023_Raw.csv.
Le dataset final est exporté sous le nom DF_2021-23_Concat_Raw.csv dans data/processed/.
"""
import pandas as pd
import os

# Répertoire où se trouvent les fichiers bruts
input_dir = "data/raw"

# Liste explicite des fichiers d'entrée correspondant à dvc.yaml
input_files = [
    f"{input_dir}/DF_2021_Raw.csv",
    f"{input_dir}/DF_2022_Raw.csv",
    f"{input_dir}/DF_2023_Raw.csv"
]

# Vérifier que tous les fichiers existent, sinon lever une erreur
missing_files = [f for f in input_files if not os.path.exists(f)]
if missing_files:
    raise FileNotFoundError(f"Les fichiers suivants sont manquants : {missing_files}")

# Afficher les fichiers sélectionnés
print("Fichiers sélectionnés pour la concaténation :")
for file in input_files:
    print(f" - {file}")

# Charger les fichiers dans une liste de DataFrames
dfs = [pd.read_csv(file) for file in input_files]

# Concaténer les DataFrames en un seul, ignorer les index et remplir les NaN avec 0
df_concat = pd.concat(dfs, axis=0, ignore_index=True).fillna(0)

# Définir le répertoire de sortie et s'assurer qu'il existe
output_dir = "data/processed"
os.makedirs(output_dir, exist_ok=True)

# Sauvegarder le fichier concaténé
output_filename = f"{output_dir}/DF_2021-23_Concat_Raw.csv"
df_concat.to_csv(output_filename, index=False)
print(f"Dataset concaténé enregistré : {output_filename}")