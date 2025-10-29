"""
Script : train.py
Objectif :
    Entraîner le modèle final à partir des meilleurs hyperparamètres trouvés
    et sauvegarder le modèle entraîné.

Entrées :
    - data/processed_data/X_train_scaled.csv
    - data/processed_data/y_train.csv
    - models/best_params.pkl

Sortie :
    - models/model.pkl
"""

import pandas as pd
import yaml
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import sys


def main(params_path: str = "params.yaml") -> None:
    # 1️⃣ Charger les paramètres globaux
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    # 2️⃣ Définir les chemins utiles
    X_train_path = Path(params["paths"]["x_train_scaled"])
    y_train_path = Path(params["paths"]["y_train"])
    best_params_path = Path(params["paths"]["best_params"])
    model_path = Path(params["paths"]["model"])

    model_path.parent.mkdir(parents=True, exist_ok=True)

    # 3️⃣ Charger les données et hyperparamètres
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()

    # Supprimer la colonne 'date' si présente
    if "date" in X_train.columns:
        print("Suppression de la colonne 'date' avant entraînement.")
        X_train = X_train.drop(columns=["date"])

    with open(best_params_path, "rb") as f:
        best_params = pickle.load(f)

    print(f"Meilleurs hyperparamètres utilisés : {best_params}")

    # 4️⃣ Créer et entraîner le modèle final
    model = RandomForestRegressor(
        random_state=params["model"]["random_state"],
        **best_params
    )

    model.fit(X_train, y_train)
    print("✅ Modèle entraîné avec succès.")

    # 5️⃣ Sauvegarder le modèle
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"✅ Modèle final sauvegardé dans : {model_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "params.yaml")
