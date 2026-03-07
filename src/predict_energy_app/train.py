import pandas as pd
import numpy as np
import bentoml
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


from predict_energy_app.preprocess import Preprocess


def train():
    # 1. Création du dossier models s'il n'existe pas
    Path("models").mkdir(exist_ok=True)

    # 2. Chargement des données brutes
    print("--- Chargement des données ---")
    data_path = "data/2016_Building_Energy_Benchmarking.csv"
    raw_df = pd.read_csv(data_path)

    # 3. Preprocessing
    print("--- Preprocessing en cours ---")
    preprocessor = Preprocess()
    df_processed = preprocessor.run_pipeline(raw_df, training=True)

    # 4. Séparation Features (X) et Target (y)
    target = "SiteEnergyUseWN(kBtu)"
    X = df_processed.drop(columns=[target])

    y_log = np.log1p(df_processed[target])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )

    # 5. Entraînement du modèle final
    print(f"--- Entraînement du modèle sur {X_train.shape[0]} lignes ---")
    rfr = RandomForestRegressor(
        max_depth=20,
        max_features="sqrt",
        min_samples_split=10,
        n_estimators=300,
        random_state=42,
    )
    rfr.fit(X_train, y_train)

    # 6. Évaluation rapide
    predictions_log = rfr.predict(X_test)

    # Pour une MAE parlante, on repasse en valeurs réelles
    predictions_real = np.expm1(predictions_log)
    y_test_real = np.expm1(y_test)

    print(
        f"RandomForest - MAE (Réelle): {mean_absolute_error(y_test_real, predictions_real):.2f} kBtu"
    )
    print(f"RandomForest - R² Score (Log): {r2_score(y_test, predictions_log):.4f}")

    # 7. Enregistrement BentoML
    # C'est une excellente idée de sauvegarder le preprocessor avec !
    bentoml.sklearn.save_model(
        "energy_regressor_rfr",
        rfr,
        custom_objects={"preprocessor": preprocessor},
        signatures={"predict": {"batchable": False}},
    )

    print("--- Modèle enregistré avec succès dans le store BentoML ! ---")


if __name__ == "__main__":
    train()
