"""
Script : gridsearch.py
Objectif :
    Chercher les meilleurs hyperparamètres d’un modèle de régression
    en utilisant GridSearchCV sur l’ensemble d’entraînement normalisé.

Entrées :
    - data/processed_data/X_train_scaled.csv
    - data/processed_data/y_train.csv

Sorties :
    - models/best_model.pkl
    - models/best_params.pkl
"""

import pandas as pd
import yaml
from pathlib import Path
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import sys


def main(params_path: str = "params.yaml") -> None:
    # 1️⃣ Charger les paramètres depuis params.yaml
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    # Récupération des chemins
    X_train_path = Path(params["paths"]["x_train_scaled"])
    y_train_path = Path(params["paths"]["y_train"])
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    # 2️⃣ Charger les données
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()  # ravel() => vecteur 1D

    print(f"Données chargées : X_train={X_train.shape}, y_train={y_train.shape}")

    # 3️⃣ Définir le modèle et la grille d’hyperparamètres
    model_type = params["model"]["type"]
    grid_params = params["grid"]

    if model_type == "RandomForestRegressor":
        model = RandomForestRegressor(random_state=params["model"]["random_state"])
    else:
        raise ValueError(f"Modèle non pris en charge : {model_type}")

    print("Début du GridSearchCV...")

    # Extraire uniquement les hyperparamètres valides pour le modèle
    param_grid = {
        "n_estimators": grid_params["n_estimators"],
        "max_depth": grid_params["max_depth"],
        "min_samples_split": grid_params["min_samples_split"],
        "min_samples_leaf": grid_params["min_samples_leaf"],
    }

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=grid_params["cv"],
        n_jobs=-1,
        scoring=grid_params["scoring"],  # <-- reste en dehors du param_grid
        verbose=2
    )


    # 4️⃣ Entraînement
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    best_params = grid.best_params_
    best_score = grid.best_score_

    print(f"Meilleurs paramètres : {best_params}")
    print(f"Score moyen (cross-val) : {best_score:.4f}")

    # 5️⃣ Sauvegarde du modèle et des hyperparamètres
    best_model_path = model_dir / "best_model.pkl"
    best_params_path = model_dir / "best_params.pkl"

    with open(best_model_path, "wb") as f:
        pickle.dump(best_model, f)
    with open(best_params_path, "wb") as f:
        pickle.dump(best_params, f)

    print(f"✅ Modèle sauvegardé dans : {best_model_path}")
    print(f"✅ Paramètres sauvegardés dans : {best_params_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "params.yaml")
