import json
import streamlit as st
import numpy as np
import pandas as pd
import xgboost as xgb
import plotly.express as px
import joblib
from scripts.data_ingestion import generate_synthetic_data
from scripts.feature_engineering import prepare_features
from scripts.model_pipeline import predict_future_recursive

@st.cache_resource
def load_metrics():
    with open('models/metrics.json', 'r') as f:
        return json.load(f)


st.set_page_config(page_title="Delivery Planner Pro", layout="wide")

@st.cache_resource
def load_model():
# En lugar de model.load_model(...)
    model = joblib.load('./models/demand_model.joblib')
    print("Modelo de pronostico cargado en memoria")
    return model

@st.cache_resource
def load_historic_data():
# En lugar de model.load_model(...)
    data = pd.read_csv('./datalake/feature_engineered_data.csv')
    print("Datos históricos cargados en memoria")
    return data

# Interfaz
st.title("🚀 Planificador de Recursos Operativos")
st.sidebar.header("Parámetros")
selected_zone = st.sidebar.selectbox("Zona", [i for i in range(5)])
future_days = st.sidebar.slider("Días a Predecir", 1, 7, 3)
lead_time = st.sidebar.slider("Tiempo Entrega (min)", 20, 60, 35)
# agregar selectbox para definir la ventana de tiempo que se muestra en el gráfico
time_window = st.sidebar.selectbox("Ventana de Tiempo en Gráfico", ["24 horas", "3 días", "7 días", '15 días'], index=0)
# agregar boton para recargar modelo y datos
if st.sidebar.button("Recargar Modelo y Datos"):
    load_model.clear()
    load_historic_data.clear()
    st.sidebar.success("Modelo y datos recargados.")

# Procesamiento de predicciones
model = load_model()

df_feat = load_historic_data()
df_future, hist = predict_future_recursive(
    model=model,
    df_historic=df_feat,
    zone_id=selected_zone,
    future_days=future_days
)

print("Columns in df_future:")
print(df_future.columns)
print("Columns in df_feat:")
print(hist.columns)
print(df_future.head(3))

# combinar históricos y predicciones
zone_data = pd.concat([hist, df_future], ignore_index=True)
zone_data['timestamp'] = pd.to_datetime(zone_data['timestamp'])
zone_data = zone_data.rename(columns={'orders': 'pred'})
# ordenar por timestamp
zone_data = zone_data.sort_values(by='timestamp').reset_index(drop=True)
print(f" Dimensiones de zone_data después de concatenar las predicciones: {zone_data.shape}")
# Cálculo de repartidores
zone_data['repartidores'] = (zone_data['pred'] * (lead_time/60) * 1.2).apply(np.ceil)

# Visualización
col1, col2 = st.columns([2, 1])
time_window_hours = {
    "24 horas": 24,
    "3 días": 72,
    "7 días": 168,
    "15 días": 360
}[time_window]
filtered_data = zone_data[zone_data['timestamp'] >= (zone_data['timestamp'].max() - pd.Timedelta(hours=time_window_hours))]
with col1:
    st.subheader(f"Pronóstico de Demanda vs Capacidad - Zona/Sector {selected_zone}")
    fig = px.line(filtered_data, x='timestamp', y=['pred', 'repartidores'], 
                  labels={'value': 'Cantidad de órdenes', 'timestamp': 'Hora', 'pred': 'Órdenes Pronosticadas', 'repartidores': 'Repartidores Necesarios'},
                  title="Órdenes vs Repartidores Necesarios")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Métricas Próximas 24h")
    next_24 = filtered_data.iloc[:24]
    st.metric("Pico de Órdenes", int(next_24['pred'].max()))
    st.metric("Máximo de Repartidores", int(next_24['repartidores'].max()))
    st.write("Horas Críticas:")
    st.dataframe(next_24.sort_values(by='pred', ascending=False)[['timestamp', 'pred']].head(5))


# --- Sección de Rendimiento del Modelo ---
# st.sidebar.divider()
# st.sidebar.subheader("🎯 Rendimiento del Modelo (Test)")

try:
    metrics = load_metrics()
    
    # Mostrar en el sidebar o en una pestaña principal
    with st.expander("Ver detalle de métricas de error"):
        c1, c2 = st.columns(2)
        c1.metric("R² Score", f"{metrics['r2']:.2f}")
        c2.metric("MAPE", f"{metrics['mape']:.1f}%")
        
        st.divider()
        
        col_a, col_b, col_c = st.columns(3)
        col_a.write(f"**MAE:** {metrics['mae']:.2f}")
        col_b.write(f"**MSE:** {metrics['mse']:.2f}")
        col_c.write(f"**RMSE:** {metrics['rmse']:.2f}")
        
        st.caption("Calculado sobre los últimos 7 días de prueba (fuera de muestra).")

except FileNotFoundError:
    st.sidebar.warning("Métricas no disponibles. Entrena el modelo.")