import bentoml
import pandas as pd
import numpy as np
from predict_energy_app.schema import EnergyInput

# On définit le nom du modèle enregistré dans le store BentoML
MODEL_TAG = "energy_regressor_rfr:latest"


@bentoml.service(name="energy_service", traffic={"timeout": 60})
class EnergyService:

    def __init__(self):
        # 1. Chargement de la référence du modèle BentoML
        self.model_ref = bentoml.sklearn.get(MODEL_TAG)

        # 2. Chargement du modèle Random Forest réel
        self.model_sklearn = self.model_ref.load_model()

        # 3. Extraction du préprocesseur sauvegardé dans les custom_objects
        self.preprocessor = self.model_ref.custom_objects["preprocessor"]

    @bentoml.api
    def predict(self, input_data: EnergyInput) -> dict:
        """
        Prédit la consommation d'énergie (kBtu) d'un bâtiment à Seattle.
        L'interface Swagger gère la validation des champs.
        """
        # Conversion des données d'entrée en DataFrame
        df = pd.DataFrame([input_data.model_dump()])

        # Prétraitement via le pipeline (calcul de l'âge, surface par étage, etc.)
        processed_df = self.preprocessor.run_pipeline(
            df, training=False
        )  # training=False => utiliser les paramètres appris

        # Prédiction (le résultat est à l'échelle logarithmique)
        prediction_log = self.model_sklearn.predict(processed_df)

        # On inverse la transformation log1p faite durant l'entraînement
        prediction_kbtu = np.expm1(prediction_log[0])

        return {
            "model_version": self.model_ref.tag.version,
            "prediction_kbtu": round(float(prediction_kbtu), 2),
            "unit": "kBtu",
        }


# Initialisation du service pour BentoML
