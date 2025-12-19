import json
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
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
time_window = st.sidebar.selectbox("Ventana de Tiempo en Gráfico", ["24 horas", "3 días", "7 días", '15 días'], index=0)

if st.sidebar.button("🔄 Recargar Modelo y Datos"):
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
zone_data['repartidores'] = (zone_data['pred'] * (lead_time/60) * 1.2).apply(np.ceil)

# --- CUERPO PRINCIPAL ---
st.title("🚀 Planificador de Recursos Operativos")

# 1. Pestañas para organizar la información
tab_forecast, tab_geo, tab_metrics = st.tabs(["📈 Pronóstico", "🗺️ Análisis Espacial", "🧠 Salud del Modelo"])

with tab_forecast:
    col1, col2 = st.columns([2, 1])
    time_window_hours = {"24 horas": 24, "3 días": 72, "7 días": 168, "15 días": 360}[time_window]
    filtered_data = zone_data[zone_data['timestamp'] >= (zone_data['timestamp'].max() - pd.Timedelta(hours=time_window_hours))]
    
    with col1:
        st.subheader(f"Demanda vs Capacidad - Zona {selected_zone}")
        fig = px.line(filtered_data, x='timestamp', y=['pred', 'repartidores'], 
                      color_discrete_map={'pred': '#00CC96', 'repartidores': '#EF553B'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Indicadores Próximas 24h")
        next_24 = filtered_data.iloc[:24]
        st.metric("Pico de Órdenes", int(next_24['pred'].max()))
        st.metric("Repartidores Necesarios", int(next_24['repartidores'].max()))
        st.write("**Horas Críticas:**")
        st.dataframe(next_24.sort_values(by='pred', ascending=False)[['timestamp', 'pred']].head(5), hide_index=True)

with tab_geo:
    st.subheader("Distribución Espacial de la Demanda")
    # Generar data de todas las zonas para el mapa (último punto predicho)
    latest_preds = []
    for z in range(5):
        z_df, _ = predict_future_recursive(model, df_feat, zone_id=z, future_days=1)
        latest_preds.append(z_df.iloc[0]) # Tomamos la hora actual
    
    map_df = pd.DataFrame(latest_preds)
    
    # Configuración de mapa 3D con Pydeck
    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(
            latitude=map_df['lat'].mean(),
            longitude=map_df['lon'].mean(),
            zoom=12,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
                'ColumnLayer',
                data=map_df,
                get_position='[lon, lat]',
                get_elevation='orders',
                elevation_scale=100,
                radius=200,
                get_fill_color='[orders * 5, 100, 150, 200]',
                pickable=True,
                auto_highlight=True,
            ),
        ],
        tooltip={"text": "Zona: {zone_id}\nÓrdenes: {orders}"}
    ))
    st.caption("La altura y color de las barras representa la intensidad de demanda actual en cada zona.")

with tab_metrics:
    try:
        metrics = load_metrics()
        st.subheader("📊 Evaluación Fuera de Muestra (Out-of-Sample)")
        st.markdown("""
        Las métricas se calculan comparando las predicciones del modelo contra datos reales que el modelo **nunca vio durante su entrenamiento**. 
        Esto garantiza que el error reportado sea una estimación honesta de su desempeño futuro.
        """)
        
        m_col1, m_col2, m_col3 = st.columns(3)
        
        with m_col1:
            st.metric("R² Score", f"{metrics['r2']:.2f}")
            st.write("**¿Qué es?** Indica qué tanto porcentaje de la variabilidad de la demanda explica el modelo. 1.0 es perfecto.")
            
            st.metric("MAPE", f"{metrics['mape']:.1f}%")
            st.write("**¿Qué es?** Error porcentual promedio. Un 10% significa que fallamos por un 10% en el volumen total.")

        with m_col2:
            st.metric("MAE", f"{metrics['mae']:.2f}")
            st.write("**¿Qué es?** Error Absoluto Medio. Es la cantidad de órdenes promedio que el modelo falla por cada hora.")
            
            st.metric("RMSE", f"{metrics['rmse']:.2f}")
            st.write("**¿Qué es?** Penaliza errores grandes. Si es mucho mayor al MAE, significa que tuvimos picos imprevistos muy grandes.")

        with m_col3:
            st.write("**Interpretación Operativa**")
            if metrics['mape'] < 15:
                st.success("✅ Modelo Altamente Confiable para planificación de turnos.")
            else:
                st.warning("⚠️ Precaución: El margen de error sugiere aumentar el 'Buffer' de repartidores.")
            
            st.info(f"**MSE:** {metrics['mse']:.2f}")

    except FileNotFoundError:
        st.error("No se encontraron métricas. Por favor, ejecuta el proceso de entrenamiento.")