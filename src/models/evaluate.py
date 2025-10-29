"""
Script : evaluate.py
Objectif :
    Évaluer le modèle final sur les données de test et sauvegarder les scores et prédictions.

Entrées :
    - models/model.pkl
    - data/processed_data/X_test_scaled.csv
    - data/processed_data/y_test.csv

Sorties :
    - metrics/scores.json
    - data/processed_data/predictions.csv
"""

import pandas as pd
import yaml
import json
import pickle
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys


def main(params_path: str = "params.yaml") -> None:
    # 1️⃣ Charger les paramètres
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    # 2️⃣ Définir les chemins
    model_path = Path(params["paths"]["model"])
    X_test_path = Path(params["paths"]["x_test_scaled"])
    y_test_path = Path(params["paths"]["y_test"])
    metrics_path = Path(params["paths"]["scores"])
    predictions_path = Path(params["paths"]["preds"])

    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    # 3️⃣ Charger le modèle et les données
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel()


    # 4️⃣ Prédictions
    y_pred = model.predict(X_test)

    # 5️⃣ Calcul des métriques
    scores = {
        "mse": round(mean_squared_error(y_test, y_pred), params["evaluate"]["round_digits"]),
        "mae": round(mean_absolute_error(y_test, y_pred), params["evaluate"]["round_digits"]),
        "r2": round(r2_score(y_test, y_pred), params["evaluate"]["round_digits"])
    }

    print("Résultats de l'évaluation :")
    for k, v in scores.items():
        print(f"  {k}: {v}")

    # 6️⃣ Sauvegarde des métriques et prédictions
    with open(metrics_path, "w") as f:
        json.dump(scores, f, indent=4)

    preds_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred
    })
    preds_df.to_csv(predictions_path, index=False)

    print(f"✅ Métriques sauvegardées dans : {metrics_path}")
    print(f"✅ Prédictions sauvegardées dans : {predictions_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "params.yaml")
