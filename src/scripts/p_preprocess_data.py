#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import os
import logging
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    filename="logs/preprocess.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def clean_data(df):
    logging.info(f"Début du prétraitement : {len(df)} lignes initiales")
    
    # Vérification initiale
    logging.info(f"Colonnes présentes : {df.columns.tolist()}")
    if 'Ft' not in df.columns:
        logging.error("Colonne 'Ft' manquante")
        raise KeyError("Colonne 'Ft' manquante")
    
    # Nettoyage de 'Ft'
    df['Ft'] = df['Ft'].str.lower()
    df = df[df['Ft'] != 'unknown']
    dico_fuel = {
        'petrol': 'Essence', 'hydrogen': 'Essence', 'e85': 'Essence', 'lpg': 'Essence',
        'ng': 'Essence', 'ng-biomethane': 'Essence', 'diesel': 'Diesel',
        'petrol/electric': 'Hybride', 'diesel/electric': 'Hybride', 'electric': 'Electrique'
    }
    df['Ft'] = df['Ft'].replace(dico_fuel)
    df = df[df['Ft'] != 'Electrique']
    logging.info(f"Nettoyage de 'Ft' : {len(df)} lignes restantes")

    # Gestion des valeurs manquantes
    required_cols = ['Ewltp (g/km)', 'Mk', 'Cn', 'Ec (cm3)', 'Ep (KW)', 'M (kg)', 'Fc', 'Erwltp (g/km)']
    df = df.dropna(subset=required_cols)
    logging.info(f"Suppression des NaN : {len(df)} lignes restantes")

    # Nettoyage de 'Mk'
    df['Mk'] = df['Mk'].str.upper()
    target_brands = ['CITROEN', 'FORD', 'FIAT', 'RENAULT', 'MERCEDES', 'BMW', 'VOLKSWAGEN', 'ALPINE', 
                     'INEOS', 'LAMBORGHINI', 'TOYOTA', 'JAGUAR', 'GREAT WALL MOTOR', 'CATERHAM', 'PEUGEOT', 
                     'MAN', 'OPEL', 'ALLIED VEHICLES', 'IVECO', 'MITSUBISHI', 'DS', 'MAZDA', 'SUZUKI', 
                     'SUBARU', 'HYUNDAI', "AUDI", "NISSAN", "SKODA", "SEAT", "DACIA", "VOLVO", "KIA",
                     "LAND ROVER", "MINI", "PORSCHE", "ALFA ROMEO", "SMART", "LANCIA", "JEEP"]
    df['Mk'] = df['Mk'].apply(lambda x: next((brand for brand in target_brands if brand in x), x))
    dico_marque = {'DS': 'CITROEN', 'VW': 'VOLKSWAGEN', '?KODA': 'SKODA', 'ŠKODA': 'SKODA',
                   'PSA AUTOMOBILES SA': 'PEUGEOT', 'FCA ITALY': 'FIAT', 'ALFA  ROMEO': 'ALFA ROMEO',
                   'LANDROVER': 'LAND ROVER'}
    df['Mk'] = df['Mk'].replace(dico_marque)
    brands_to_delete = ['TRIPOD', 'API CZ', 'MOTO STAR', 'REMOLQUES RAMIREZ', 'AIR-BRAKES', 
                        'SIN MARCA', 'WAVECAMPER', 'CASELANI', 'PANDA']
    df = df[~df['Mk'].isin(brands_to_delete)]
    filtered_brands = [brand for brand in df['Mk'].unique() if df['Mk'].tolist().count(brand) >= 5]
    df = df[df['Mk'].isin(filtered_brands)]
    logging.info(f"Nettoyage de 'Mk' : {len(df)} lignes restantes")

    # Suppression des doublons
    df = df.drop_duplicates()
    logging.info(f"Suppression des doublons : {len(df)} lignes restantes")

    # Détection des outliers
    def detect_outliers(df, target_col, group_cols=["Cn", "Ft", "Year"]):
        stats = df.groupby(group_cols)[target_col].mean().reset_index(name=f'{target_col}_mean')
        df_merged = pd.merge(df, stats, on=group_cols, how="left")
        df_merged[f"diff_{target_col}"] = (df_merged[target_col] - df_merged[f"{target_col}_mean"]).abs()
        q1, q3 = df_merged[f"diff_{target_col}"].quantile([0.25, 0.75])
        iqr = q3 - q1
        seuil = q3 + 1.5 * iqr
        outliers = df_merged[df_merged[f"diff_{target_col}"] >= seuil]
        df_clean = df_merged[df_merged[f"diff_{target_col}"] <= seuil]
        logging.info(f"Outliers pour {target_col} (seuil {seuil:.1f}) : {len(outliers)} supprimés, {len(df_clean)} lignes restantes")
        return df_clean

    for col in ['Ewltp (g/km)', 'Fc', 'M (kg)', 'Ec (cm3)', 'Ep (KW)', 'Erwltp (g/km)']:
        df = detect_outliers(df, col)
    
    df = df[df['Ft'] != 'Hybride']
    logging.info(f"Suppression des hybrides : {len(df)} lignes restantes")

    # Encodage des variables catégoriques
    df = pd.get_dummies(df, columns=['Ft', 'Mk'], prefix=['Ft', 'Mk'], drop_first=False)
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)
    logging.info("Encodage des variables catégoriques terminé")
    
    return df

def main():
    input_file = "data/processed/DF_2021-23_Concat_Raw.csv"
    if not os.path.exists(input_file):
        logging.error(f"Fichier d'entrée manquant : {input_file}")
        raise FileNotFoundError(f"Fichier d'entrée manquant : {input_file}")
    
    logging.info("Début du prétraitement")
    df = pd.read_csv(input_file, low_memory=False)
    df_clean = clean_data(df)
    
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "DF_2021-23_Processed.csv"
    df_clean.to_csv(output_file, index=False)
    logging.info(f"Prétraitement terminé, fichier enregistré dans {output_file}")

if __name__ == "__main__":
    main()