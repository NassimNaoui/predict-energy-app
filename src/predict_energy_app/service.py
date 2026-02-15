import bentoml
import pandas as pd
from predict_energy_app.schema import EnergyInput


@bentoml.service(name="energy_service")
class EnergyService:
    
    def __init__(self):
        # On charge les modèles réels (les objets Sklearn)
        # C'est ici qu'on extrait le modèle de l'objet Bento
        self.lr_model_obj = bentoml.sklearn.get("energy_regressor_lr:latest")
        self.lr_model_sklearn = self.lr_model_obj.load_model()
        
        self.rf_model_obj = bentoml.sklearn.get("energy_regressor_rf:latest")
        self.rf_model_sklearn = self.rf_model_obj.load_model()

    @bentoml.api
    def predict_linear(self, input_data: EnergyInput) -> dict:
        # On passe le modèle sklearn ET l'objet bento (pour le preprocessor)
        result = self._run_prediction(input_data, self.lr_model_obj, self.lr_model_sklearn)
        return {"model": "LinearRegression", "prediction_kbtu": result}

    @bentoml.api
    def predict_random_forest(self, input_data: EnergyInput) -> dict:
        result = self._run_prediction(input_data, self.rf_model_obj, self.rf_model_sklearn)
        return {"model": "RandomForest", "prediction_kbtu": result}

    def _run_prediction(self, input_data: EnergyInput, bento_model, sklearn_model):
        df = pd.DataFrame([input_data.model_dump()])
        
        # Le preprocessor est dans les custom_objects de l'objet BENTO
        preprocessor = bento_model.custom_objects['preprocessor']
        
        # Pipeline
        processed_df = preprocessor.run_pipeline(df, training=False)
        
        # Prédiction sur le modèle SKLEARN
        prediction = sklearn_model.predict(processed_df)
        return float(prediction[0])