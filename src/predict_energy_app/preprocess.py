import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

class Preprocess():
    def __init__(self):
        order_year = [['1900-1944', '1945-1973', '1974-1999', '2000-2016']]
        order_floors = [['<= 1', '2', '3', '4', '5', '> 5']]
        
        self.enc_year = OrdinalEncoder(categories=order_year)
        self.enc_floors = OrdinalEncoder(categories=order_floors)
        self.ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform="pandas")
        # AJOUT DU SCALER
        self.scaler = StandardScaler()
        self.num_cols = ['Latitude', 'Longitude', 'PropertyGFATotal']

    def define_scope(self, df):
        non_residential_building = ['Hotel', 'K-12 School', 'University', 'Small- and Mid-Sized Office',
                                    'Self-Storage Facility', 'Warehouse', 'Large Office','Medical Office', 
                                    'Retail Store','Hospital', 'Distribution Center','Worship Facility',
                                    'Senior Care Community','Supermarket / Grocery Store', 'Laboratory',
                                    'Refrigerated Warehouse', 'Restaurant']
        
        df_filtered = df[df['PrimaryPropertyType'].isin(non_residential_building)].copy()
        df_filtered = df_filtered[df_filtered['Outlier'].isna()]
        return df_filtered

    def run_pipeline(self, df, training=True):
        # 1. Scope
        if training:
            df = self.define_scope(df)
            df = df[(df['SiteEnergyUseWN(kBtu)'] < 1.5e8) & (df['TotalGHGEmissions'] < 5e3)].copy()

        # 2. Features Engineering
        df['NumberofBuildings'] = np.where(df['NumberofBuildings'] == 0, 1, df['NumberofBuildings'])
        df['building_sup_1'] = (df['NumberofBuildings'] == 1).astype(int)
        
        df['category_year_built'] = np.where(df['YearBuilt'] < 1945, '1900-1944',
                                  np.where(df['YearBuilt'] < 1974, '1945-1973', 
                                  np.where(df['YearBuilt'] < 2000, '1974-1999','2000-2016')))
        
        bins = [-1, 1, 2, 3, 4, 5, float('inf')]
        labels = ['<= 1', '2', '3', '4', '5', '> 5']
        df['nb_floors_category'] = pd.cut(df['NumberofFloors'], bins=bins, labels=labels).astype(str)

        # 3. Encodages & Scaling
        if training:
            self.enc_year.fit(df[['category_year_built']])
            self.enc_floors.fit(df[['nb_floors_category']])
            self.ohe.fit(df[['PrimaryPropertyType']])
            self.scaler.fit(df[self.num_cols])

        # Transformation (appliquée dans les deux cas)
        df['year_built_encoded'] = self.enc_year.transform(df[['category_year_built']])
        df['nb_floors_encoded'] = self.enc_floors.transform(df[['nb_floors_category']])
        
        encoded_type = self.ohe.transform(df[['PrimaryPropertyType']])
        # On remplace les valeurs numériques par leurs versions scalées
        df[self.num_cols] = self.scaler.transform(df[self.num_cols])

        # 4. Assemblage final
        # On définit les colonnes numériques finales (maintenant scalées)
        cols_base = self.num_cols + ['building_sup_1', 'year_built_encoded', 'nb_floors_encoded']
        
        if training:
            # On crée un nouveau DF pour éviter les problèmes de colonnes résiduelles
            final_df = pd.concat([df[cols_base + ['SiteEnergyUseWN(kBtu)']], encoded_type], axis=1)
        else:
            final_df = pd.concat([df[cols_base], encoded_type], axis=1)
            
        return final_df