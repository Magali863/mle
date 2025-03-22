#!/usr/bin/env python
# coding: utf-8
import requests
import urllib.parse
import numpy as np
import pandas as pd
import os
import json
import logging
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    filename="logs/import.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def fetch_data(table_list):
    records = []
    for table in table_list:
        query = f"""
        SELECT DISTINCT [Year], Mk, Cn, [M (kg)], [Ewltp (g/km)], Ft, [Ec (cm3)], [Ep (KW)], [Erwltp (g/km)], Fc
        FROM [CO2Emission].[latest].[{table}]
        WHERE Mk IS NOT NULL AND Cn IS NOT NULL AND [M (kg)] IS NOT NULL
        AND [Ewltp (g/km)] IS NOT NULL AND Ft IS NOT NULL AND [Ec (cm3)] IS NOT NULL
        AND [Ep (KW)] IS NOT NULL AND [Erwltp (g/km)] IS NOT NULL AND [Year] IS NOT NULL AND Fc IS NOT NULL
        """
        encoded_query = urllib.parse.quote(query)
        page = 1
        logging.info(f"Récupération des données pour la table {table}")
        while True:
            url = f"https://discodata.eea.europa.eu/sql?query={encoded_query}&p={page}&nrOfHits=100000"
            response = requests.get(url)
            if response.status_code != 200:
                logging.error(f"Erreur API pour {table}, page {page}: {response.status_code}")
                raise Exception(f"Erreur API: {response.status_code}")
            data = response.json()
            new_records = data.get("results", [])
            if not new_records:
                break
            records.extend(new_records)
            page += 1
        logging.info(f"{len(records)} enregistrements récupérés pour {table}")
    return pd.DataFrame(records)

def clean_data(df):
    # Suppression des doublons (sans 'Cn' et 'Year')
    subset_cols = [col for col in df.columns if col not in ['Cn', 'Year']]
    df = df.drop_duplicates(subset=subset_cols)
    logging.info(f"Suppression des doublons : {len(df)} lignes restantes")

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
    return df[['Mk', 'Cn', 'M (kg)', 'Ewltp (g/km)', 'Ft', 'Ec (cm3)', 'Ep (KW)', 'Erwltp (g/km)', 'Year', 'Fc']]

def main():
    table_list = ['co2cars_2021Pv23', 'co2cars_2022Pv25', 'co2cars_2023Pv27']
    logging.info("Début de l'importation des données")
    df = fetch_data(table_list)
    df_clean = clean_data(df)
    
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    for year in [2021, 2022, 2023]:
        df_year = df_clean[df_clean['Year'] == year]
        output_file = output_dir / f"DF_{year}_Raw.csv"
        df_year.to_csv(output_file, index=False)
        logging.info(f"Données pour {year} enregistrées dans {output_file}")
    
    metadata = {"files": {str(year): f"data/raw/DF_{year}_Raw.csv" for year in [2021, 2022, 2023]}}
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)
    logging.info("Importation terminée avec succès")

if __name__ == "__main__":
    main()