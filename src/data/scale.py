"""
Script : scale.py
Objectif :
    Normaliser les colonnes numériques des données d'entraînement et de test,
    tout en conservant la colonne 'date' non modifiée.

Entrées :
    - data/processed_data/X_train.csv
    - data/processed_data/X_test.csv

Sorties :
    - data/processed_data/X_train_scaled.csv
    - data/processed_data/X_test_scaled.csv
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
import yaml
from pathlib import Path
import sys


def main(params_path: str = "params.yaml") -> None:
    # 1️⃣ Charger les paramètres depuis params.yaml
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    # Récupérer les chemins d'entrée / sortie
    X_train_path = Path(params["paths"]["x_train"])
    X_test_path = Path(params["paths"]["x_test"])
    X_train_scaled_path = Path(params["paths"]["x_train_scaled"])
    X_test_scaled_path = Path(params["paths"]["x_test_scaled"])

    # Options de scaling
    scale_mean = params["scale"]["with_mean"]
    scale_std = params["scale"]["with_std"]

    # 2️⃣ Charger les fichiers CSV
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)

    print(f"Colonnes initiales : {list(X_train.columns)}")

    # 3️⃣ Identifier les colonnes à exclure (non numériques)
    exclude_cols = ["date"] if "date" in X_train.columns else []
    numeric_cols = X_train.columns.difference(exclude_cols)

    print(f"Colonnes à normaliser : {list(numeric_cols)}")
    print(f"Colonnes exclues : {list(exclude_cols)}")

    # 4️⃣ Normaliser uniquement les colonnes numériques
    scaler = StandardScaler(with_mean=scale_mean, with_std=scale_std)
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train[numeric_cols]),
        columns=numeric_cols
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test[numeric_cols]),
        columns=numeric_cols
    )

    # 5️⃣ Réintégrer les colonnes exclues (date)
    if exclude_cols:
        for col in exclude_cols:
            X_train_scaled.insert(0, col, X_train[col].values)
            X_test_scaled.insert(0, col, X_test[col].values)

    # 6️⃣ Sauvegarder les fichiers normalisés
    X_train_scaled_path.parent.mkdir(parents=True, exist_ok=True)
    X_train_scaled.to_csv(X_train_scaled_path, index=False)
    X_test_scaled.to_csv(X_test_scaled_path, index=False)

    print(f"✅ Données normalisées sauvegardées :\n - {X_train_scaled_path}\n - {X_test_scaled_path}")
    print(f"Dimensions : X_train_scaled = {X_train_scaled.shape}, X_test_scaled = {X_test_scaled.shape}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "params.yaml")
