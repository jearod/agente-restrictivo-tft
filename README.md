# 🛡️ Agente Restrictivo TFT: Prevención de Day Trading Subóptimo

Este repositorio contiene la implementación algorítmica de un **Agente Restrictivo** basado en aprendizaje profundo. Su objetivo es transformar la predicción pasiva de los mercados financieros en una intervención mecánica activa, bloqueando operaciones subóptimas para proteger a los inversores minoristas.

## 📊 Origen de los Datos
Los datos históricos de precios y volumen utilizados para entrenar y validar este modelo han sido extraídos de Kaggle: **[Historical ETF Data](https://www.kaggle.com/datasets/liewyousheng/historical-etf)**. 
Para garantizar la reproducibilidad exacta del experimento (Grado R1), un subconjunto consolidado de estos datos se encuentra empaquetado y versionado directamente en este repositorio dentro del archivo `data/etf.zip`.

## 🧠 Arquitectura del Modelo
El núcleo predictivo utiliza la arquitectura **Temporal Fusion Transformer (TFT)** a través de la librería `pytorch-forecasting`. El modelo procesa:
* **Metadatos Estáticos:** Identificadores de los activos (Tickers).
* **Variables Temporales Conocidas:** Índices de calendario (Día de la semana, Mes).
* **Variables Temporales Desconocidas:** Histórico de precios (Open, High, Low, Close) y Volumen.

## 🌍 Corpus Global y Transfer Learning
El proyecto divide el aprendizaje y la validación en dos grandes fases para evitar la fuga de datos (*Data Leakage*):
1. **Fase de Pre-entrenamiento (`data/pretrain/`):** El modelo aprende las "reglas universales" del mercado usando 12 ETFs globales representativos (VOO, QQQ, DIA, IEUS, EWJV, etc.).
2. **Fase de Inferencia Masiva (`data/raw/` extraído del zip):** Se aplica *Zero-Shot Transfer Learning* sobre más de 500 ETFs adicionales que el modelo jamás ha visto, demostrando su capacidad de generalización para reducir el Maximum Drawdown.

## 📁 Estructura Principal del Proyecto
* **`containers/`**: Contiene la infraestructura base (`docker-compose.yml`) para levantar MLflow y Jenkins.
* **`data/`**: Almacena el corpus versionado (`etf.zip`) y los resultados de inferencia (`processed/`).
* **`models/best/`**: Registro automático donde se promueve y congela el mejor modelo entrenado (`best_model.ckpt`) para su uso en producción/inferencia masiva.
* **`pipeline/`**: Definición del flujo CI/CD (`Jenkinsfile`).
* **`src/`**: Código fuente de Machine Learning (preprocesamiento, entrenamiento, evaluación e inferencia masiva).
* **`Dockerfile`**: Receta en la raíz del workspace para construir el entorno hermético de experimentación.

## ⚙️ CI/CD y Reproducibilidad R1 (Jenkins & Docker)
Este proyecto no requiere configuración local de librerías. Utiliza una arquitectura MLOps **Docker-out-of-Docker (DooD)** orquestada por Jenkins, garantizando un entorno 100% hermético, inmutable y reproducible. El pipeline se encarga de extraer los datos, entrenar la red neuronal, promover al mejor modelo a `models/best/` y ejecutar la inferencia masiva.

### 🚀 Instrucciones de Uso

Para ejecutar el experimento de investigación completo desde tu servidor de CI/CD:

1. Levanta la infraestructura base apuntando a la carpeta de contenedores:
   ```bash
   docker-compose -f containers/docker-compose.yml up -d
2. Entra a tu panel de Jenkins (http://localhost:8080) y configura un nuevo Pipeline apuntando a este repositorio.

3. Haz clic en "Build with Parameters".

4. Selecciona tus parámetros:

   * TRAIN_MODEL: Entrena el modelo desde cero y auto-promueve el mejor a models/best/.

   * RUN_MASS_INFERENCE: Ejecuta la validación sobre los 500+ ETFs usando el modelo de la carpeta best.

5. Al finalizar, descarga tu reporte mass_inference_results.csv o tu best_model.ckpt desde la pestaña de Artefactos en Jenkins.