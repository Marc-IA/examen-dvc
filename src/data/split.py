"""
Script : split.py
But : Séparer le dataset en ensembles d'entraînement et de test.

Étapes :
1. Charger les chemins et paramètres depuis params.yaml
2. Lire le fichier de données brutes
3. Séparer les features (X) et la cible (y)
4. Effectuer le split train/test selon les paramètres
5. Sauvegarder les fichiers dans data/processed_data/

Ce script dépend de la sortie du script get_data.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import sys
from pathlib import Path


def main(params_path: str = "params.yaml") -> None:
    """Crée les ensembles d'entraînement et de test à partir des données brutes."""

    # Charger les paramètres
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    # Récupération des chemins et paramètres
    raw_path = Path(params["paths"]["raw_csv"])
    target_col = params["data"]["target"]
    test_size = params["split"]["test_size"]
    random_state = params["split"]["random_state"]
    shuffle = params["split"]["shuffle"]

    # Lecture du dataset brut
    df = pd.read_csv(raw_path)
    print(f"Lecture du dataset : {raw_path.name} ({df.shape[0]} lignes)")

    # Supprimer la colonne 'date' si elle existe
    if "date" in df.columns:
        print("Suppression de la colonne 'date' : non utilisée pour la modélisation.")
        df = df.drop(columns=["date"])

    # Séparation features / target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=shuffle
    )

    # Définition des chemins de sortie
    X_train_path = Path(params["paths"]["x_train"])
    X_test_path = Path(params["paths"]["x_test"])
    y_train_path = Path(params["paths"]["y_train"])
    y_test_path = Path(params["paths"]["y_test"])

    # Création des dossiers de sortie
    X_train_path.parent.mkdir(parents=True, exist_ok=True)

    # Sauvegarde des fichiers
    X_train.to_csv(X_train_path, index=False)
    X_test.to_csv(X_test_path, index=False)
    y_train.to_csv(y_train_path, index=False)
    y_test.to_csv(y_test_path, index=False)

    print("Split terminé.")
    print(f"X_train : {X_train.shape}, X_test : {X_test.shape}")
    print(f"y_train : {y_train.shape}, y_test : {y_test.shape}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "params.yaml")
