import json
import subprocess
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
# import pydeck as pdk
import joblib
from scripts.model_pipeline import predict_future_recursive

# --- CARGA DE RECURSOS ---
@st.cache_resource
def load_metrics():
    with open('models/metrics.json', 'r') as f:
        return json.load(f)

@st.cache_resource
def load_model():
    model = joblib.load('./models/demand_model.joblib')
    return model

@st.cache_resource
def load_historic_data():
    data = pd.read_csv('./datalake/feature_engineered_data.csv')
    return data

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Delivery Planner Pro", layout="wide", page_icon="🚴")

# --- SIDEBAR ---
st.sidebar.header("🕹️ Panel de Control")
selected_zone = st.sidebar.selectbox("Zona Operativa", [i for i in range(5)])
future_days = st.sidebar.slider("Días a Predecir", 1, 7, 3)
lead_time = st.sidebar.slider("Tiempo Entrega (min)", 20, 60, 35)
sequrity_margin = st.sidebar.slider("Margen de Seguridad (%)", 0, 50, 20)
time_window = st.sidebar.selectbox("Ventana de Tiempo en Gráfico", ['15 días', "7 días", "3 días", "24 horas", "12 horas"], index=0)

if st.sidebar.button("🔄 Recargar Modelo y Datos"):
    subprocess.run(["python", "scripts/model_pipeline.py"])
    load_model.clear()
    load_historic_data.clear()
    st.sidebar.success("Modelo y datos actualizados.")

# --- LÓGICA DE PREDICCIÓN ---
model = load_model()
df_feat = load_historic_data()
df_future, hist = predict_future_recursive(
    model=model,
    df_historic=df_feat,
    zone_id=selected_zone,
    future_days=future_days
)

# Preparación de datos consolidados
zone_data = pd.concat([hist, df_future], ignore_index=True)
zone_data['timestamp'] = pd.to_datetime(zone_data['timestamp'])
zone_data = zone_data.rename(columns={'orders': 'pred'})
zone_data = zone_data.sort_values(by='timestamp').reset_index(drop=True)
zone_data['repartidores'] = (zone_data['pred'] * (lead_time/60) * (1+sequrity_margin/100)).apply(np.ceil)

# --- CUERPO PRINCIPAL ---
st.title("🚀 Planificador de Recursos Operativos")
tab_forecast, tab_metrics = st.tabs(["📈 Pronóstico", "🧠 Salud del Modelo"])

with tab_forecast:
    col1, col2 = st.columns([2, 1])
    time_window_hours = {
        "24 horas": 24,
        "12 horas": 12,
        "3 días": 72,
        "7 días": 168,
        "15 días": 360}[time_window]
    filtered_data = zone_data[zone_data['timestamp'] >= (zone_data['timestamp'].max() - pd.Timedelta(hours=time_window_hours))]
    
    with col1:
        st.subheader(f"Demanda vs Capacidad - Zona {selected_zone}")
        fig = px.line(filtered_data, x='timestamp', y=['pred', 'repartidores'], 
                      color_discrete_map={'pred': '#00CC96', 'repartidores': '#EF553B'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader(f"Indicadores Próximas {24*future_days}h")
        next_24 = filtered_data.iloc[-24*future_days:]
        st.metric("Pico de Órdenes", int(next_24['pred'].max()))
        st.metric("Repartidores Necesarios", int(next_24['repartidores'].max()))
        st.write("**Horas Críticas:**")
        st.dataframe(next_24.sort_values(by='pred', ascending=False)[['timestamp', 'pred']].head(5), hide_index=True)
with tab_metrics:
    try:
        metrics = load_metrics()
        st.subheader("📊 Evaluación Fuera de Muestra (Out-of-Sample)")
        st.markdown("""
        Las métricas se calculan comparando las predicciones del modelo contra datos datos reales de los últimos 15 días del histórico observado que el modelo **nunca vio durante su entrenamiento**. 
        Esto garantiza que el error reportado sea una estimación honesta de su desempeño futuro.
        """)
        
        m_col1, m_col2, m_col3 = st.columns(3)
        
        with m_col1:
            st.metric("R² Score", f"{metrics['r2']:.2f}")
            st.write("El modelo explica el {:.1f}% de la variabilidad en la demanda.".format(metrics['r2'] * 100))
            
            st.metric("MAPE", f"{metrics['mape']:.1f}%")
            st.write("El modelo tiene un error porcentual absoluto medio del {:.1f}%.".format(metrics['mape']))

        with m_col2:
            st.metric("MAE", f"{metrics['mae']:.2f}")
            st.write("El modelo se desvía en promedio {:.2f} órdenes.".format(metrics['mae']))
            
            st.metric("RMSE", f"{metrics['rmse']:.2f}")
            st.write("El error cuadrático medio es de {:.2f} órdenes.".format(metrics['rmse']))

        with m_col3:
            st.write("**Interpretación Operativa**")
            if metrics['mape'] < 15:
                st.success("✅ Modelo Altamente Confiable para planificación de turnos.")
            else:
                st.warning("⚠️ Precaución: El margen de error sugiere aumentar el 'Buffer' de repartidores.")
            
            st.info(f"**MSE:** {metrics['mse']:.2f}")

    except FileNotFoundError:
        st.error("No se encontraron métricas. Por favor, ejecuta el proceso de entrenamiento.")