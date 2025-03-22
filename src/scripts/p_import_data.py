#!/usr/bin/env python
# coding: utf-8

# Importation des librairies pour la requête SQL et l'enregistrement du fichier final :
import requests
import urllib.parse

# Importation des librairies classiques :
import numpy as np
import pandas as pd

# Importation des librairies spécifiques à l'enregistrement du fichier final : 
import os
import json

# Liste des tables disponibles sur le site de l'Agence Européenne à date :
table_list = ['co2cars_2021Pv23', 'co2cars_2022Pv25', 'co2cars_2023Pv27']

# Définition de la requête et boucle pour l'appliquer à toutes les tables :
records = []

for table in table_list:
    query = f"""
    SELECT DISTINCT [Year] AS Year, Mk, Cn, [M (kg)], [Ewltp (g/km)], Ft, [Ec (cm3)], [Ep (KW)], [Erwltp (g/km)], Fc
    FROM [CO2Emission].[latest].[{table}]
    WHERE Mk IS NOT NULL 
      AND Cn IS NOT NULL 
      AND [M (kg)] IS NOT NULL
      AND [Ewltp (g/km)] IS NOT NULL
      AND Ft IS NOT NULL
      AND [Ec (cm3)] IS NOT NULL
      AND [Ep (KW)] IS NOT NULL
      AND [Erwltp (g/km)] IS NOT NULL
      AND [Year] IS NOT NULL
      AND Fc IS NOT NULL
    """
    # Encodage de la requête pour l'inclure dans l'URL :
    encoded_query = urllib.parse.quote(query)

    # Initialisation :
    page = 1

    # Boucle while pour parcourir toutes les pages de l'API :
    while True:
        url = f"https://discodata.eea.europa.eu/sql?query={encoded_query}&p={page}&nrOfHits=100000"
        response = requests.get(url)
        data = response.json()
        new_records = data.get("results", [])
        if not new_records:
            break
        records.extend(new_records)
        page += 1

# Transformation en DataFrame :
df = pd.DataFrame(records)

# Suppression des doublons potentiels à travers les années (sans "Cn") :
subset_cols = [col for col in df.columns if col not in ['Cn', 'Year']]
df = df.drop_duplicates(subset=subset_cols)

# Vérification de la colonne "Ft" : travail de catégorisation nécessaire
df['Ft'] = df['Ft'].str.lower()
df = df[df['Ft'] != 'unknown']

# Rassemblement des variables
dico_fuel = {
    'petrol': 'Essence',
    'hydrogen': 'Essence',
    'e85': 'Essence',
    'lpg': 'Essence',
    'ng': 'Essence',
    'ng-biomethane': 'Essence',
    'diesel': 'Diesel',
    'petrol/electric': 'Hybride',
    'diesel/electric': 'Hybride',
    'electric': 'Electrique'
}
df['Ft'] = df['Ft'].replace(dico_fuel)
df = df[df['Ft'] != 'Electrique']

# Traitement de la colonne "Mk" :
df['Mk'] = df['Mk'].str.upper()

target_brands = [
    'CITROEN', 'FORD', 'FIAT', 'RENAULT', 'MERCEDES', 'BMW', 'VOLKSWAGEN', 'ALPINE', 
    'INEOS', 'LAMBORGHINI', 'TOYOTA', 'JAGUAR', 'GREAT WALL MOTOR', 'CATERHAM', 'PEUGEOT', 
    'MAN', 'OPEL', 'ALLIED VEHICLES', 'IVECO', 'MITSUBISHI', 'DS', 'MAZDA', 'SUZUKI', 
    'SUBARU', 'HYUNDAI', "AUDI", "NISSAN", "SKODA", "SEAT", "DACIA", "VOLVO", "KIA",
    "LAND ROVER", "MINI", "PORSCHE", "ALFA ROMEO", "SMART", "LANCIA", "JEEP"
]

def extract_brand(value):
    for brand in target_brands:
        if brand in value:
            return brand
    return value
df['Mk'] = df['Mk'].apply(extract_brand)

dico_marque = {
    'DS': 'CITROEN',
    'VW': 'VOLKSWAGEN',
    '?KODA': 'SKODA',
    'ŠKODA': 'SKODA',
    'PSA AUTOMOBILES SA': 'PEUGEOT',
    'FCA ITALY': 'FIAT',
    'ALFA  ROMEO': 'ALFA ROMEO',
    'LANDROVER': 'LAND ROVER'
}
df['Mk'] = df['Mk'].replace(dico_marque)

brands_to_delete = [
    'TRIPOD', 'API CZ', 'MOTO STAR', 'REMOLQUES RAMIREZ', 'AIR-BRAKES', 
    'SIN MARCA', 'WAVECAMPER', 'CASELANI', 'PANDA'
]
df = df[~df['Mk'].isin(brands_to_delete)]

def filter_brands(df, col='Mk', threshold=5):
    brands = df[col].tolist()
    unique_brands = df[col].unique().tolist()
    filtered_brands = [brand for brand in unique_brands if brands.count(brand) >= threshold]
    return filtered_brands

filtered_brands = filter_brands(df, col='Mk', threshold=5)
df = df[df['Mk'].isin(filtered_brands)]

# Détection des valeurs aberrantes
def detect_outliers(df, target_col, group_cols=["Cn", "Ft", "Year"]):
    stats = (
        df.groupby(group_cols)
          .agg(**{f'{target_col}_mean': (target_col, 'mean')})
          .reset_index()
    )
    df_merged = pd.merge(df, stats, on=group_cols, how="left")
    diff_col = f"diff_{target_col}"
    df_merged[diff_col] = (df_merged[target_col] - df_merged[f"{target_col}_mean"]).abs()
    q1 = df_merged[diff_col].quantile(0.25)
    q3 = df_merged[diff_col].quantile(0.75)
    iqr = q3 - q1
    seuil = (q3 + 1.5 * iqr).round(1)
    nb_outliers = len(df_merged[df_merged[diff_col] >= seuil])
    print(f'Nombre de lignes dont la valeur de "{target_col}" dépasse le seuil de {seuil}: {nb_outliers}')
    df_clean_no_outliers = df_merged[df_merged[diff_col] <= seuil]
    print(f"Nombre de lignes après suppression des outliers : {len(df_clean_no_outliers)}")
    return df_clean_no_outliers

columns_to_filter = ['Ewltp (g/km)', 'Fc', 'M (kg)', 'Ec (cm3)', 'Ep (KW)', 'Erwltp (g/km)']
df_temp = df.copy()

for col in columns_to_filter:
    print(col)
    df_temp = detect_outliers(df_temp, col)

print("\nAprès filtrage successif, le nombre de lignes restantes est de :", len(df_temp))
df_clean_no_outliers_final = df_temp

df_clean_no_outliers_final = df_clean_no_outliers_final[['Mk', 'Cn', 'M (kg)', 'Ewltp (g/km)', 'Ft', 'Ec (cm3)', 
                                                         'Ep (KW)', 'Erwltp (g/km)', 'Year', 'Fc']]

df_clean_no_outliers_final = df_clean_no_outliers_final[df_clean_no_outliers_final['Ft'] != 'Hybride']

# Encodage des variables catégorielles
df_clean_no_outliers_final = pd.get_dummies(df_clean_no_outliers_final, columns=['Ft'], prefix='Ft', drop_first=False)
bool_cols = df_clean_no_outliers_final.select_dtypes(include=['bool']).columns
df_clean_no_outliers_final[bool_cols] = df_clean_no_outliers_final[bool_cols].astype(int)

df_clean_no_outliers_final = pd.get_dummies(df_clean_no_outliers_final, columns=['Mk'], prefix='Mk', drop_first=False)
bool_cols = df_clean_no_outliers_final.select_dtypes(include=['bool']).columns
df_clean_no_outliers_final[bool_cols] = df_clean_no_outliers_final[bool_cols].astype(int)

# Enregistrement des fichiers par année avec des noms fixes
output_dir = "data/raw"
os.makedirs(output_dir, exist_ok=True)

# Séparer les données par année et enregistrer avec des noms fixes
for year in [2021, 2022, 2023]:
    df_year = df_clean_no_outliers_final[df_clean_no_outliers_final['Year'] == year]
    output_filename = f"{output_dir}/DF_{year}_Raw.csv"
    df_year.to_csv(output_filename, index=False)
    print(f"Données brutes pour {year} enregistrées dans : {output_filename}")

# Enregistrement des métadonnées
metadata_file = f"{output_dir}/metadata.json"
metadata = {
    "files": {
        "2021": "data/raw/DF_2021_Raw.csv",
        "2022": "data/raw/DF_2022_Raw.csv",
        "2023": "data/raw/DF_2023_Raw.csv"
    }
}
with open(metadata_file, "w") as f:
    json.dump(metadata, f)