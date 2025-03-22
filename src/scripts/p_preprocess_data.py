#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import os

# Charger le dataset concaténé
input_file = "data/raw/DF_2021-23_Concat_Raw.csv"
df = pd.read_csv(input_file, low_memory=False)

# Vérification initiale
print(df.head(10))
print(df.info())
print(df.describe())
print(df.isna().mean().round(5) * 100)

# Traitement de "Ft"
df['Ft'] = df['Ft'].str.lower()
df = df[df['Ft'] != 'unknown']
dico_fuel = {
    'petrol': 'Essence', 'hydrogen': 'Essence', 'e85': 'Essence', 'lpg': 'Essence',
    'ng': 'Essence', 'ng-biomethane': 'Essence', 'diesel': 'Diesel',
    'petrol/electric': 'Hybride', 'diesel/electric': 'Hybride', 'electric': 'Electrique'
}
df['Ft'] = df['Ft'].replace(dico_fuel)
df = df[df['Ft'] != 'Electrique']

# Gestion des valeurs manquantes
df = df.dropna(subset=['Ewltp (g/km)', 'Mk', 'Cn', 'ec (cm3)', 'ep (KW)', 'm (kg)', 'Fuel consumption ', 'Erwltp (g/km)'])

# Traitement de "Mk"
df['Mk'] = df['Mk'].str.upper()
target_brands = ['CITROEN', 'FORD', 'FIAT', 'RENAULT', 'MERCEDES', 'BMW', 'VOLKSWAGEN', 'ALPINE',
                 'INEOS', 'LAMBORGHINI', 'TOYOTA', 'JAGUAR', 'GREAT WALL MOTOR', 'CATERHAM', 'PEUGEOT',
                 'MAN', 'OPEL', 'ALLIED VEHICLES', 'IVECO', 'MITSUBISHI', 'DS', 'MAZDA', 'SUZUKI',
                 'SUBARU', 'HYUNDAI', "AUDI", "NISSAN", "SKODA", "SEAT", "DACIA", "VOLVO", "KIA",
                 "LAND ROVER", "MINI", "PORSCHE", "ALFA ROMEO", "SMART", "LANCIA", "JEEP"]
def extract_brand(value):
    for brand in target_brands:
        if brand in value:
            return brand
    return value
df['Mk'] = df['Mk'].apply(extract_brand)
dico_marque = {
    'DS': 'CITROEN', 'VW': 'VOLKSWAGEN', '?KODA': 'SKODA', 'ŠKODA': 'SKODA',
    'PSA AUTOMOBILES SA': 'PEUGEOT', 'FCA ITALY': 'FIAT', 'ALFA  ROMEO': 'ALFA ROMEO',
    'LANDROVER': 'LAND ROVER'
}
df['Mk'] = df['Mk'].replace(dico_marque)
brands_to_delete = ['TRIPOD', 'API CZ', 'MOTO STAR', 'REMOLQUES RAMIREZ', 'AIR-BRAKES',
                    'SIN MARCA', 'WAVECAMPER', 'CASELANI', 'PANDA']
df = df[~df['Mk'].isin(brands_to_delete)]
def filter_brands(df, col='Mk', threshold=5):
    brands = df[col].tolist()
    unique_brands = df[col].unique().tolist()
    filtered_brands = [brand for brand in unique_brands if brands.count(brand) >= threshold]
    return filtered_brands
filtered_brands = filter_brands(df, col='Mk', threshold=5)
df = df[df['Mk'].isin(filtered_brands)]

# Suppression des doublons
df_clean = df.drop_duplicates()

# Détection et suppression des outliers
def detect_outliers(df, target_col, group_cols=["Cn", "Ft", "year"]):
    stats = df.groupby(group_cols).agg(**{f'{target_col}_mean': (target_col, 'mean')}).reset_index()
    df_merged = pd.merge(df, stats, on=group_cols, how="left")
    diff_col = f"diff_{target_col}"
    df_merged[diff_col] = (df_merged[target_col] - df_merged[f"{target_col}_mean"]).abs()
    q1 = df_merged[diff_col].quantile(0.25)
    q3 = df_merged[diff_col].quantile(0.75)
    iqr = q3 - q1
    seuil = (q3 + 1.5 * iqr).round(1)
    nb_outliers = len(df_merged[df_merged[diff_col] >= seuil])
    print(f"Outliers pour {target_col} (seuil {seuil}): {nb_outliers}")
    df_clean_no_outliers = df_merged[df_merged[diff_col] <= seuil]
    print(f"Lignes restantes : {len(df_clean_no_outliers)}")
    return df_clean_no_outliers

columns_to_filter = ['Ewltp (g/km)', 'Fuel consumption ', 'm (kg)', 'ec (cm3)', 'ep (KW)', 'Erwltp (g/km)']
df_temp = df_clean.copy()
for col in columns_to_filter:
    df_temp = detect_outliers(df_temp, col)
df_clean_no_outliers_final = df_temp
df_clean_no_outliers_final = df_clean_no_outliers_final[['Mk', 'Cn', 'm (kg)', 'Ewltp (g/km)', 'Ft', 'ec (cm3)',
                                                         'ep (KW)', 'Erwltp (g/km)', 'year', 'Fuel consumption ']]

# Mise de côté des hybrides
df_clean_no_outliers_final = df_clean_no_outliers_final[df_clean_no_outliers_final['Ft'] != 'Hybride']

# Encodage des variables catégoriques
df_clean_no_outliers_final = pd.get_dummies(df_clean_no_outliers_final, columns=['Ft', 'Mk'], prefix=['Ft', 'Mk'], drop_first=False)
bool_cols = df_clean_no_outliers_final.select_dtypes(include=['bool']).columns
df_clean_no_outliers_final[bool_cols] = df_clean_no_outliers_final[bool_cols].astype(int)

# Enregistrement
output_dir = "data/processed"
os.makedirs(output_dir, exist_ok=True)
output_filename = f"{output_dir}/DF_2021-23_Processed.csv"
df_clean_no_outliers_final.to_csv(output_filename, index=False)
print(f"Dataset pré-traité : {output_filename}")