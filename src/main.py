import os
import warnings
from pytorch_forecasting import TimeSeriesDataSet

# Importar nuestros módulos creados previamente
from src.data_pipeline import load_and_sort_data, strict_temporal_partition
from src.tft_dataset import add_time_features, create_tft_dataset
from src.train_model import train_tft_model

# Desactivar advertencias de PyTorch para mantener la consola limpia
warnings.filterwarnings("ignore")

def main():
    print("=== Iniciando el Pipeline del Agente Restrictivo ===")
    
    # 1. Definir la ruta del archivo de datos (basado en la estructura de carpetas)
    # Asumimos que el script se ejecuta desde la raíz del proyecto
    data_path = "data/raw/etf_sample.csv"
    
    if not os.path.exists(data_path):
        print(f"\n[ERROR] No se encontró el archivo de datos en: {data_path}")
        print("Por favor, asegúrate de colocar un archivo CSV de un ETF allí para continuar.")
        return

    # 2. Ingestión y Partición Temporal
    print("\n--- Fase 1: Ingestión y Partición Estricta ---")
    df = load_and_sort_data(data_path)
    train_df, val_df, test_df = strict_temporal_partition(df)
    
    # 3. Preprocesamiento: Ingeniería de características
    print("\n--- Fase 2: Preprocesamiento de Tensores ---")
    print("Agregando índices de tiempo y metadatos estáticos...")
    train_df = add_time_features(train_df)
    val_df = add_time_features(val_df)
    # El test_df lo prepararemos más adelante cuando hagamos inferencia pura
    
    # 4. Creación del Dataset para el TFT
    print("\n--- Fase 3: Construcción de TimeSeriesDataSet ---")
    print("Construyendo el tensor de entrenamiento...")
    training_dataset = create_tft_dataset(train_df)
    
    print("Construyendo el tensor de validación...")
    # En PyTorch Forecasting, la validación se crea a partir del dataset de entrenamiento
    # para asegurar que los mapeos categóricos y escaladores sean exactamente los mismos
    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, val_df, predict=True, stop_randomization=True
    )
    
    # 5. Entrenamiento y Seguimiento con MLflow
    print("\n--- Fase 4: Entrenamiento y Transfer Learning ---")
    # Invocamos el script que conecta con MLflow y entrena el modelo
    # Reducimos el batch_size a 32 por defecto para evitar sobrecargar la memoria RAM/VRAM
    model, trainer = train_tft_model(training_dataset, validation_dataset, batch_size=32)
    
    print("\n=== Pipeline completado con éxito ===")
    print("El modelo ha sido entrenado y guardado.")
    print("Puedes revisar las métricas gráficas ejecutando 'mlflow ui' en tu terminal.")

if __name__ == "__main__":
    main()