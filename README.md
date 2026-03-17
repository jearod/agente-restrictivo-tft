# 🛡️ Agente Restrictivo TFT: Prevención de Day Trading Subóptimo

Este repositorio contiene la implementación algorítmica de un **Agente Restrictivo** basado en aprendizaje profundo. Su objetivo es transformar la predicción pasiva de los mercados financieros en una intervención mecánica activa, bloqueando operaciones subóptimas para proteger a los inversores minoristas.

## 🧠 Arquitectura del Modelo
El núcleo predictivo utiliza la arquitectura **Temporal Fusion Transformer (TFT)** a través de la librería `pytorch-forecasting`. El modelo procesa:
* **Metadatos Estáticos:** Identificadores de los activos (Tickers).
* **Variables Temporales Conocidas:** Índices de calendario (Mes, Día de la semana).
* **Variables Temporales Desconocidas:** Histórico de precios (Open, High, Low, Close) y Volumen.

## 🌍 Corpus Global y Transfer Learning
Para lograr una generalización robusta sin sufrir de sobreajuste local, el modelo se pre-entrena utilizando Transfer Learning sobre un corpus global de los ETFs más representativos de tres mercados principales:
* **América:** S&P 500 (VOO), Nasdaq (QQQ), Dow Jones (DIA)
* **Europa:** Eurostoxx 50 (FEZ), DAX (EWG), CAC 40 (EWQ), FTSE 100 (EWU), IBEX 35 (EWP), FTSE MIB (EWI)
* **Asia:** Nikkei 225 (EWJ), Hang Seng (EWH), Kospi (EWY), Shanghai Composite (MCHI)

*Nota: El pipeline de datos aplica un Protocolo Estricto de Partición Temporal (70% Entrenamiento, 15% Validación, 15% Prueba) de forma independiente para cada activo, garantizando cero contaminación de datos futuros.*

## ⚙️ Requisitos y MLOps
El proyecto está diseñado para cumplir con el grado de reproducibilidad **R1 (Experiment Reproducibility)**.
* **Aislamiento:** Docker y Docker Compose.
* **Seguimiento de Experimentos:** MLflow (Métricas, Parámetros y Modelos).

## 🚀 Instrucciones de Instalación y Uso

1. **Clonar el repositorio y preparar los datos:**
   Asegúrate de colocar los archivos `.csv` históricos de los 14 ETFs mencionados en la carpeta `data/raw/`.

2. **Levantar el entorno reproducible:**
   El proyecto utiliza Docker Compose para levantar el orquestador de entrenamiento y el servidor de MLflow en paralelo, manteniendo la persistencia de datos mediante volúmenes.
   ```bash
   docker-compose up --build