
---

# 🚴 Delivery Planner Pro: Sistema de Inteligencia Operativa

**Delivery Planner Pro** es una solución avanzada de Machine Learning diseñada para optimizar la gestión de repartidores en empresas de delivery de última milla. El sistema utiliza modelos predictivos para anticipar la demanda por zona y calcular automáticamente los recursos necesarios, permitiendo una planificación operativa proactiva y basada en datos.

## 📋 Descripción del Proyecto

El núcleo de esta solución es un motor de pronóstico que transforma datos históricos de órdenes en planes de trabajo accionables. El sistema aborda el problema crítico del desbalance entre repartidores y órdenes, optimizando los costos operativos y garantizando niveles de servicio superiores.

### Características Principales

* **Pronóstico de Demanda Parametrizable:** Permite al usuario definir el horizonte de predicción (de 1 a 7 días) según las necesidades tácticas o estratégicas.
* **Inferencia Recursiva:** Implementa una lógica de predicción donde el modelo utiliza sus propios resultados previos como entrada para pasos futuros, asegurando la continuidad en horizontes largos.
* **Cálculo Automático de Recursos:** Traduce órdenes predichas en cantidad de repartidores requeridos, considerando tiempos de entrega y márgenes de seguridad.
* **Dashboard Interactivo:** Interfaz visual construida en Streamlit con mapas 3D y análisis de métricas de salud del modelo.

## 🏗️ Estructura del Proyecto

El repositorio sigue una arquitectura modular para facilitar el mantenimiento y la escalabilidad:

```text
MUTTDATA-TECHNICAL-TEST/
├── datalake/                # Almacenamiento de datos crudos y procesados
│   ├── demand_data.csv
│   └── feature_engineered_data.csv
├── models/                  # Modelos persistidos y métricas de evaluación
│   ├── demand_model.joblib
│   └── metrics.json
├── scripts/                 # Módulos de procesamiento y lógica de ML
│   ├── __init__.py
│   ├── data_ingestion.py    # Ingesta y generación de datos
│   ├── feature_engineering.py # Transformaciones y creación de lags
│   └── model_pipeline.py    # Entrenamiento y lógica recursiva
├── app.py                   # Aplicación principal de Streamlit
├── requirements.txt         # Dependencias del proyecto
└── vercel.json              # Configuración de despliegue

```

## 🛠️ Stack Tecnológico

* **Lenguaje:** Python 3.9+
* **Modelado:** XGBoost Regressor
* **Ingeniería de Datos:** Pandas, Numpy
* **Visualización:** Plotly
* **Dashboard:** Streamlit

## 🧠 Supuestos Críticos de la Solución

Para garantizar la confiabilidad de los resultados, el sistema opera bajo los siguientes supuestos:

* **Estacionalidad:** Se asume que la demanda tiene patrones cíclicos horarios y semanales capturables mediante variables trigonométricas y lags.
* **Independencia de Zonas:** Cada zona operativa se modela considerando sus particularidades geográficas y volumen de datos.
* **Validación:** El desempeño se mide bajo métricas "fuera de muestra" (MAPE, MAPE, MAE) para asegurar precisión en entornos de producción reales.

## 🚀 Instalación y Uso Local

1. **Clonar el repositorio:**
```bash
git clone https://github.com/jhonflook/muttdata-technical-test.git
cd muttdata-technical-test

```

2. **Entrenamiento del modelo:**
Ejecutar una unica vez para generar los pesos iniciales del modelo:
```bash
python scripts/model_pipeline.py

```

3. **Instalar dependencias:**
```bash
pip install -r requirements.txt

```


4. **Ejecutar el Dashboard:**
```bash
streamlit run app.py

```


5. **Reentrenamiento (Opcional):**
Puedes actualizar el modelo directamente desde la interfaz de la app o ejecutando:
```bash
python scripts/model_pipeline.py

```



---
