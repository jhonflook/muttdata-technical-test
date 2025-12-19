import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def prepare_features(
        df, 
        test_days=15, 
        future_task=False, 
        future_days=7, 
        zone_id=None,
        lags=[1, 24, 168]
        ):

    def add_time_features(df):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        # Transformación cíclica para capturar que la hora 23 está cerca de la 00
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/23)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/23)
        return df

    if future_task:
        # Crear target futuro (órdenes en 'future_days' días)
        # engineered_df = pd.read_csv('./datalake/feature_engineered_data.csv')
        engineered_df = df[df['zone_id'] == zone_id].copy()
        # organizar por timestamp
        engineered_df = engineered_df.sort_values(by='timestamp').reset_index(drop=True)
        print("✅ Features históricos cargados para zona ", zone_id)
        print(engineered_df.shape)
        
        max_date = engineered_df['timestamp'].max()
        future_dates = pd.date_range(start=max_date, periods=future_days*24, freq='H')
        future_df = pd.DataFrame({
            'timestamp': future_dates,
            'zone_id': zone_id        # agregar lags desde engineered_df eficientemente
        })
        future_df = add_time_features(future_df)
        lags_cols = [
            f'lag_{l}' for l in lags
        ]
        # 3. MAPEO DINÁMICO DE LAGS (La clave de la corrección)
        for l in lags:
            # Calculamos para cada fila del futuro, qué momento exacto fue hace 'l' horas
            future_df['lookup_time'] = future_df['timestamp'] - pd.to_timedelta(l, unit='h')
            
            # Cruzamos (merge) con el histórico para traer el valor real de 'orders' en ese momento
            future_df = pd.merge(
                future_df, 
                engineered_df[['timestamp', 'orders']], 
                left_on='lookup_time', 
                right_on='timestamp', 
                how='left', 
                suffixes=('', '_hist')
            )
            
            # Renombramos y limpiamos
            future_df[f'lag_{l}'] = future_df['orders_hist']
            future_df.drop(columns=['lookup_time', 'timestamp_hist', 'orders_hist'], inplace=True)
            
            # Si el lag cae en un punto que aún no ha pasado (predicción recursiva),
            # aquí se suele usar ffill() o una constante para no tener NaNs
            future_df[f'lag_{l}'] = future_df[f'lag_{l}'].fillna(method='ffill').fillna(0)

        # 4. Latitud y Longitud constantes
        future_df['lat'] = engineered_df['lat'].iloc[-1]
        future_df['lon'] = engineered_df['lon'].iloc[-1]
        
        df = df.dropna()
        print("✅ Features preparados para tarea futura.")
        return df

    else:
        df = add_time_features(df)
        # Lags: ¿Qué pasó hace 1 hora, 1 día y 1 semana?
        for l in lags:
            df[f'lag_{l}'] = df.groupby('zone_id')['orders'].shift(l)
        # Save df before split
        df.to_csv('./datalake/feature_engineered_data.csv', index=False)
        print("✅ Features preparados y dataset guardado en datalake/feature_engineered_data.csv")
        cut_off_date = df['timestamp'].max() - timedelta(days=test_days)
        
        train_df = df[df['timestamp'] <= cut_off_date].copy()
        test_df = df[df['timestamp'] > cut_off_date].copy()
        
        # Mensaje de éxito
        print(f"✅ Dataset generado con éxito.")
        print(f"   - Split: Train hasta {cut_off_date.date()} | Test últimos {test_days} días")
        print(f"   - Registros: Train ({len(train_df)}) / Test ({len(test_df)})")
        
        return train_df, test_df        

    # return df.dropna()