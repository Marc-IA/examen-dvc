"""
Script : get_data.py
But : Télécharger le dataset depuis une URL et le sauvegarder localement.

Étapes :
1. Charger les paramètres depuis params.yaml (URL, chemin de sortie)
2. Télécharger le dataset à partir de l’URL
3. Sauvegarder le fichier CSV dans data/raw_data/raw.csv

Ce script est la première étape de la pipeline DVC.
"""

import pandas as pd
import yaml
import sys
from pathlib import Path


def main(params_path: str = "params.yaml") -> None:
    """Télécharge le dataset et le sauvegarde localement."""

    # Charger les paramètres à partir du fichier YAML
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    # Extraire l'URL et le chemin de sortie
    data_url = params["data"]["url"]
    output_path = Path(params["paths"]["raw_csv"])

    # Créer le dossier s’il n’existe pas
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Charger les données depuis l’URL
    df = pd.read_csv(data_url)

    # Sauvegarder le dataset brut localement
    df.to_csv(output_path, index=False)

    print(f"Dataset téléchargé depuis : {data_url}")
    print(f"Fichier sauvegardé sous : {output_path.resolve()}")
    print(f"Taille du dataset : {df.shape[0]} lignes, {df.shape[1]} colonnes")


if __name__ == "__main__":
    # Permet de passer un chemin alternatif de params.yaml (utile pour DVC)
    main(sys.argv[1] if len(sys.argv) > 1 else "params.yaml")
