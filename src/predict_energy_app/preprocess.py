import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class Preprocess:
    def __init__(self):
        self.ohe = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        ).set_output(transform="pandas")
        # On stocke les médianes pour éviter le data leakage
        self.medians_ = {}

    def define_scope(self, df):
        residential_building = [
            "Mixed Use Property",
            "Residence Hall",
            "High-Rise Multifamily",
            "Low-Rise Multifamily",
            "Other",
            "Office",
            "Mid-Rise Multifamily",
        ]
        return df[~df["PrimaryPropertyType"].isin(residential_building)].copy()

    def run_pipeline(self, df, training=True):
        # 0. Copie pour éviter de modifier le DF original
        df = df.copy()

        # 1. Scope (Uniquement en training)
        if training:
            df = self.define_scope(df)

        # Reset index indispensable pour le concat final
        df = df.reset_index(drop=True)

        # 2. Features Engineering

        # --- GESTION DE LA CIBLE (Uniquement si présente) ---
        target_col = "SiteEnergyUseWN(kBtu)"

        if target_col in df.columns:
            # Nettoyage des zéros
            df[target_col] = df[target_col].replace(0, np.nan)

            if training:
                # On apprend les médianes sur le train
                self.medians_ = (
                    df.groupby("PrimaryPropertyType")[target_col].median().to_dict()
                )

            # Application des médianes pour remplir les trous
            df[target_col] = df[target_col].fillna(
                df["PrimaryPropertyType"].map(self.medians_)
            )

        # Retraitements numériques des features
        df["NumberofBuildings"] = df["NumberofBuildings"].replace(0, np.nan).fillna(1)

        # On s'assure que NumberofFloors est au moins 1 pour éviter division par zéro
        df["NumberofFloors"] = df["NumberofFloors"].fillna(0) + 1

        df["Building_Age"] = 2016 - df["YearBuilt"]

        # Calculs des variables créées
        df["Mean_Surface_Per_Floor"] = (
            df["PropertyGFATotal"] / df["NumberofFloors"]
        ).round(2)

        # Calcul du ratio de parking
        df["Parking_Ratio"] = (df["PropertyGFAParking"] / df["PropertyGFATotal"]).round(
            4
        ) * 100

        # 3. Encodage
        if training:
            self.ohe.fit(df[["PrimaryPropertyType"]])

        encoded_type = self.ohe.transform(df[["PrimaryPropertyType"]]).reset_index(
            drop=True
        )

        # 4. Assemblage final
        cols_base = [
            "PropertyGFATotal",
            "Building_Age",
            "Mean_Surface_Per_Floor",
            "Parking_Ratio",
        ]

        # On ne garde la cible que si on est en train
        features_to_keep = cols_base + (
            [target_col] if target_col in df.columns else []
        )

        final_df = pd.concat([df[features_to_keep], encoded_type], axis=1)

        return final_df
