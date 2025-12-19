import json
import os
import sys
import numpy as np
import pandas as pd
from datetime import timedelta
# from pyexpat import model
import xgboost as xgb

# set raiz del proyecto con sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from scripts.data_ingestion import generate_synthetic_data
from scripts.feature_engineering import prepare_features
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def run_training():
    print("Iniciando entrenamiento...")
    raw_data = generate_synthetic_data()
    train_df, test_df = prepare_features(df=raw_data)

    target = 'orders'
    to_drop = ['timestamp', 'lat', 'lon']
    X_train = train_df.drop(to_drop+[target], axis=1)
    y_train = train_df[target]

    y_test = test_df[target]
    X_test = test_df.drop(to_drop+[target], axis=1)
    
    # Entrenar modelo
    model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6)
    model.fit(X_train, y_train)
    
    # Guardar persistencia

#    model.save_model('./models/demand_model.json')
 #   print("Modelo guardado en models/demand_model.json")
    import joblib

# ... después del entrenamiento ...
# En lugar de model.save_model(...)
    joblib.dump(model, './models/demand_model.joblib')
    print("Modelo guardado correctamente con Joblib.")
    preds = model.predict(X_test)
    
    # 3. Calcular Métricas Fuera de Muestra
    metrics = {
        "r2": float(r2_score(y_test, preds)),
        "mse": float(mean_squared_error(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mae": float(mean_absolute_error(y_test, preds)),
        "mape": float(calculate_mape(y_test, preds))
    }
    
    # 4. Guardar métricas en un JSON
    with open('./models/metrics.json', 'w') as f:
        json.dump(metrics, f)
if __name__ == "__main__":
    run_training()


def predict_future_recursive(model, df_historic, zone_id, future_days=7, lags=[1, 24, 168]):
    """
    Genera predicciones paso a paso usando pronósticos previos como lags.
    """
    # 1. Preparar el histórico base para la zona
    hist = df_historic[df_historic['zone_id'] == zone_id].copy()
    hist['timestamp'] = pd.to_datetime(hist['timestamp'])
    hist = hist.sort_values('timestamp')
    
    lat, lon = hist['lat'].iloc[-1], hist['lon'].iloc[-1]
    
    # 2. Crear el esqueleto del futuro (hora por hora)
    max_date = hist['timestamp'].max()
    future_range = pd.date_range(start=max_date + timedelta(hours=1), 
                                periods=future_days * 24, freq='H')
    
    # Este DataFrame crecerá conforme predigamos
    results = hist[['timestamp', 'zone_id', 'orders', 'lat', 'lon']].copy()
    
    print(f"🔮 Iniciando predicción recursiva para {future_days} días...")

    for current_time in future_range:
        # a. Crear fila para el momento actual
        current_row = pd.DataFrame({
            'timestamp': [current_time],
            'zone_id': [zone_id],
            'lat': [lat],
            'lon': [lon]
        })
        
        # b. Ingeniería de tiempo cíclica
        hour = current_time.hour
        current_row['hour'] = hour
        current_row['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        current_row['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        current_row['day_of_week'] = current_time.dayofweek
        
        # c. BUSCAR LAGS (Aquí ocurre la magia)
        # Buscamos en 'results' que contiene tanto datos reales como predicciones previas
        for l in lags:
            target_time = current_time - timedelta(hours=l)
            # Buscamos el valor de la orden en ese momento exacto
            past_value = results.loc[results['timestamp'] == target_time, 'orders']
            
            if not past_value.empty:
                current_row[f'lag_{l}'] = past_value.values[0]
            else:
                # Si por alguna razón no hay dato (gap), usamos el último disponible
                current_row[f'lag_{l}'] = results['orders'].iloc[-1]
        
        # d. PREDECIR
        # Preparamos X (debe tener las mismas columnas que el entrenamiento)
        # Nota: Asegúrate de que el orden de columnas z_Zone_X coincida con el entrenamiento
        X_input = current_row.drop(columns=['timestamp', 'lat', 'lon'])[
            model.feature_names_in_
        ]
        
        # Ajuste de One-Hot Encoding si es necesario para que el modelo no falle
        # (Aquí podrías usar el transformador guardado o dummies manuales)
        
        prediction = model.predict(X_input)[0]
        prediction = max(0, prediction) # No podemos tener órdenes negativas
        
        # e. INSERTAR RESULTADO EN EL DATASET PARA LA SIGUIENTE ITERACIÓN
        current_row['orders'] = prediction
        results = pd.concat([results, current_row], ignore_index=True)
        # results = pd.concat([
        #     results, 
        #     current_row[['timestamp', 'zone_id', 'orders', 'lat', 'lon']],
        #     current_row[[f'lag_{l}' for l in lags]]
        #     ], ignore_index=True)

    # Retornar solo la parte del futuro
    return results[results['timestamp'] > max_date], hist