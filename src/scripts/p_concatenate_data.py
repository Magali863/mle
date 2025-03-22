#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import os
import logging
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    filename="logs/concatenate.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def main():
    input_files = [
        "data/raw/DF_2021_Raw.csv",
        "data/raw/DF_2022_Raw.csv",
        "data/raw/DF_2023_Raw.csv"
    ]
    logging.info("Début de la concaténation")
    
    missing_files = [f for f in input_files if not os.path.exists(f)]
    if missing_files:
        logging.error(f"Fichiers manquants : {missing_files}")
        raise FileNotFoundError(f"Fichiers manquants : {missing_files}")
    
    logging.info("Fichiers sélectionnés : " + ", ".join(input_files))
    dfs = [pd.read_csv(f, low_memory=False) for f in input_files]
    df_concat = pd.concat(dfs, ignore_index=True).fillna(0)
    logging.info(f"Concaténation réussie : {len(df_concat)} lignes")
    
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "DF_2021-23_Concat_Raw.csv"
    df_concat.to_csv(output_file, index=False)
    logging.info(f"Dataset concaténé enregistré dans {output_file}")

if __name__ == "__main__":
    main()