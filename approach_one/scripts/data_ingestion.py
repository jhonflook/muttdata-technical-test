import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def generate_synthetic_data(num_zones=5, days=90):
    # 1. Configuración de tiempo
    start_date = datetime.now() - timedelta(days=days)
    date_range = pd.date_range(start=start_date, periods=days*24, freq='H')
    
    # 2. Generación de Coordenadas Espaciales (Estilo Barrios)
    # Punto central simulado (ej. Centro de una ciudad)
    center_lat, center_lon = 4.6097, -74.0817 
    zone_coords = {}
    for i in range(num_zones):
        # Variación pequeña para que parezcan barrios (aprox 1-3 km)
        zone_coords[i] = {
            'lat': center_lat + np.random.uniform(-0.02, 0.02),
            'lon': center_lon + np.random.uniform(-0.02, 0.02)
        }

    data = []
    for zone_id in range(num_zones):
        base_demand = np.random.randint(10, 50)
        lat = zone_coords[zone_id]['lat']
        lon = zone_coords[zone_id]['lon']
        
        for ts in date_range:
            # Estacionalidad
            hour_effect = 15 * np.exp(-(ts.hour - 13)**2 / 4) + 25 * np.exp(-(ts.hour - 20)**2 / 6)
            day_effect = 1.5 if ts.weekday() >= 4 else 1.0
            noise = np.random.poisson(5)
            
            demand = int((base_demand + hour_effect) * day_effect + noise)
            data.append([ts, zone_id, lat, lon, demand])
            
    df = pd.DataFrame(data, columns=['timestamp', 'zone_id', 'lat', 'lon', 'orders'])
    
    # # Mensaje de éxito
    print(f"✅ Dataset generado con éxito.")
    print(f"   - Zonas: {num_zones} | Días totales: {days}")
    return df