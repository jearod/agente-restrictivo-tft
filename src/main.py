import os
import warnings
from pytorch_forecasting import TimeSeriesDataSet
from src.data_pipeline import load_global_portfolio
from src.tft_dataset import add_time_features, create_tft_dataset
from src.train_model import train_tft_model

warnings.filterwarnings("ignore")

def main():
    print("=== Iniciando el Pipeline del Agente Restrictivo (Transfer Learning Global) ===")
    
    # 1. Ingestión Masiva y Partición
    print("\n--- Fase 1: Ingestión del Corpus Global ---")
    try:
        train_df, val_df, test_df = load_global_portfolio(data_dir="data/raw")
    except Exception as e:
        print(f"Error crítico: {e}")
        return

    # 2. Preprocesamiento: Ingeniería de características
    print("\n--- Fase 2: Preprocesamiento de Tensores ---")
    print("Agregando índices de tiempo de forma agrupada...")
    # Agrupamos por Symbol para crear un time_idx continuo para CADA activo
    train_df = train_df.groupby('Symbol', group_keys=False).apply(add_time_features)
    val_df = val_df.groupby('Symbol', group_keys=False).apply(add_time_features)
    
    # 3. Creación del Dataset para el TFT
    print("\n--- Fase 3: Construcción de TimeSeriesDataSet ---")
    training_dataset = create_tft_dataset(train_df)
    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, val_df, predict=True, stop_randomization=True
    )
    
    # 4. Entrenamiento y Seguimiento con MLflow
    print("\n--- Fase 4: Entrenamiento ---")
    model, trainer = train_tft_model(training_dataset, validation_dataset, batch_size=64)
    
    print("\n=== Pre-entrenamiento Global completado con éxito ===")

if __name__ == "__main__":
    main()