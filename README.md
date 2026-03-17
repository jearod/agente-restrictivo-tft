# 🛡️ Agente Restrictivo TFT: Prevención de Day Trading Subóptimo

Este repositorio contiene la implementación algorítmica de un **Agente Restrictivo** basado en aprendizaje profundo. Su objetivo es transformar la predicción pasiva de los mercados financieros en una intervención mecánica activa, bloqueando operaciones subóptimas para proteger a los inversores minoristas.

## 🧠 Arquitectura del Modelo
El núcleo predictivo utiliza la arquitectura **Temporal Fusion Transformer (TFT)** a través de la librería `pytorch-forecasting`. El modelo procesa:
* **Metadatos Estáticos:** Identificadores de los activos (Tickers).
* **Variables Temporales Conocidas:** Índices de calendario.
* **Variables Temporales Desconocidas:** Histórico de precios (Open, High, Low, Close) y Volumen.

## 🌍 Corpus Global y Transfer Learning
Para lograr una generalización robusta sin sufrir de sobreajuste local, el modelo se pre-entrena utilizando Transfer Learning sobre un corpus global de 12 ETFs confirmados:
* **América:** `VOO` (S&P 500), `QQQ` (Nasdaq), `DIA` (Dow Jones), `IJH` (Mid-Cap), `IJR` (Small-Cap).
* **Europa:** `IEUS` (Europa Small-Cap), `EWL` (Suiza), `EWD` (Suecia).
* **Asia / Pacífico:** `EWJV` (Japón), `VPL` (Pacífico), `CNYA` (China), `SMIN` (India).

## ⚙️ Arquitectura de Ejecución (Híbrida)
Para maximizar el rendimiento y mantener la trazabilidad (Grado R1), el proyecto utiliza un modelo híbrido:
1. **Infraestructura de Tracking:** Contenedor Docker dedicado al servidor de **MLflow** con volúmenes persistentes.
2. **Motor de Entrenamiento:** Ejecución nativa en el entorno Python (WSL/Linux).

## 🧩 Descripción de los Módulos (`src/`)

El código está estructurado de manera modular para separar las responsabilidades del pipeline de Machine Learning:

* **`data_pipeline.py` (Ingesta y Partición):** Se encarga de leer los archivos `.csv` crudos, asignar sus metadatos estáticos (el *Ticker*) y aplicar una partición temporal estricta (70% Entrenamiento, 15% Validación, 15% Prueba) de manera independiente por activo para evitar contaminación de datos futuros.
* **`tft_dataset.py` (Ingeniería de Tensores):** Transforma los DataFrames de Pandas en el formato complejo `TimeSeriesDataSet` requerido por el modelo TFT. Aquí se definen los codificadores, las variables categóricas/continuas y las ventanas de tiempo (look-back y predicción).
* **`train_model.py` (Entrenamiento y Tracking):** Define la arquitectura matemática del Temporal Fusion Transformer, configura la parada temprana (Early Stopping) para evitar el sobreajuste y conecta el ciclo de entrenamiento con el servidor local de MLflow para registrar métricas en tiempo real.
* **`evaluate.py` (Inferencia y Backtesting):** Carga el modelo guardado con mejor rendimiento e infiere sobre el 15% de datos de prueba nunca vistos. Convierte la predicción de precios en señales direccionales de trading y calcula métricas financieras críticas como el **F1-Score Direccional** y el **Maximum Drawdown (MDD)** para validar la eficacia del Agente.
* **`main.py` (Orquestador):** El script maestro que importa e invoca de forma secuencial los scripts anteriores (pipeline de datos -> dataset -> entrenamiento) centralizando la ejecución del proyecto.

## 🚀 Instrucciones de Instalación y Uso

**1. Preparar los datos:**
Asegúrate de colocar los archivos `.csv` históricos en la carpeta `data/raw/`.

**2. Levantar el Servidor de MLflow (Docker en background):**
```bash
docker-compose up -d