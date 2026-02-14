import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from src.predict_energy_app.preprocess import Preprocess

def train():
    # 1. Création du dossier models s'il n'existe pas
    Path("models").mkdir(exist_ok=True)

    # 2. Chargement des données brutes
    print("--- Chargement des données ---")
    data_path = 'data/2016_Building_Energy_Benchmarking.csv'
    raw_df = pd.read_csv(data_path)

    # 3. Preprocessing
    print("--- Preprocessing en cours ---")
    preprocessor = Preprocess()
    # On utilise training=True pour fiter les encodeurs
    df_processed = preprocessor.run_pipeline(raw_df, training=True)

    # 4. Séparation Features (X) et Target (y)
    target = 'SiteEnergyUseWN(kBtu)'
    X = df_processed.drop(columns=[target])
    y = df_processed[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Entraînement du modèle
    print(f"--- Entraînement des modèles sur {X_train.shape[0]} lignes ---")
    rfr = RandomForestRegressor(n_estimators=100, random_state=42)
    rfr.fit(X_train, y_train)

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    models = [rfr, lr]

    # 6. Évaluation rapide
    for model in models:
        predictions = model.predict(X_test)
        model_name = type(model).__name__
        print(f"{model_name} - MAE: {mean_absolute_error(y_test, predictions):.2f}")
        print(f"{model_name} - R² Score: {r2_score(y_test, predictions):.2f}")


if __name__ == "__main__":
    train()