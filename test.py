import json
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from scripts.data_ingestion import generate_synthetic_data
from scripts.feature_engineering import prepare_features
from scripts.model_pipeline import predict_future_recursive

selected_zone = 2
functure_days = 7
lead_time = 30
def load_metrics():
    with open('models/metrics.json', 'r') as f:
        return json.load(f)

def load_model():
# En lugar de model.load_model(...)
    model = joblib.load('models/demand_model.joblib')
    print("Modelo de pronostico cargado en memoria")
    return model

# Procesamiento de predicciones
model = load_model()
print("Modelo cargado para predicciones.")
# data = generate_synthetic_data(days=8) # Traemos datos recientes para generar los lag
df_feat = pd.read_csv('./datalake/feature_engineered_data.csv')
print("Datos históricos cargados para construir los lagas para las predicciones futuras.")
df_future = predict_future_recursive(
    model=model,
    df_historic=df_feat,
    zone_id=selected_zone,
    future_days=functure_days
)
print("Columns in df_future:")
print(df_future.columns)
print("Columns in df_feat:")
print(df_feat.columns)
print(df_future.head(3))

df_future = df_future.rename(columns={'orders': 'pred'})
zone_data = pd.concat([df_feat, df_future], ignore_index=True)
print(f" Dimensiones de zone_data después de concatenar las predicciones: {zone_data.shape}")
# Cálculo de repartidores
df_future['repartidores'] = (df_future['pred'] * (lead_time/60) * 1.2).apply(np.ceil)