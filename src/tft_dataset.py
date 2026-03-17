import pandas as pd
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet

def add_time_features(df, symbol="SPY"):
    """
    Prepara el DataFrame agregando las columnas categóricas y de tiempo
    requeridas por la arquitectura TFT.
    """
    df = df.copy()
    
    # 1. Identificador de grupo (Static Metadata)
    # Si tienes múltiples ETFs, esta columna los diferenciará
    if 'Symbol' not in df.columns:
        df['Symbol'] = symbol
        
    # 2. Índice de tiempo continuo (obligatorio para PyTorch Forecasting)
    # Asumimos que los datos ya vienen ordenados del script anterior
    df['time_idx'] = np.arange(len(df))
    
    # 3. Variables temporales conocidas (Known Future Inputs)
    # Extraemos el mes y el día de la semana como características numéricas
    df['Month'] = df['Date'].dt.month.astype(str)
    df['DayOfWeek'] = df['Date'].dt.dayofweek.astype(str)
    
    return df

def create_tft_dataset(train_df, max_prediction_length=5, max_encoder_length=30):
    """
    Convierte el DataFrame de entrenamiento en un TimeSeriesDataSet,
    mapeando la topología de datos según el diseño del Agente Restrictivo.
    """
    # Configuramos el dataset siguiendo los lineamientos de la arquitectura TFT
    training_dataset = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="Close", # Asignamos el precio de cierre como target inicial para pre-entrenamiento
        group_ids=["Symbol"], # Metadatos estáticos
        
        # Longitud de la ventana de contexto (pasado) y predicción (futuro)
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        
        # Variables categóricas estáticas (Static Metadata)
        static_categoricals=["Symbol"],
        
        # Variables categóricas que cambian con el tiempo pero se conocen en el futuro
        time_varying_known_categoricals=["Month", "DayOfWeek"],
        time_varying_known_reals=["time_idx"],
        
        # Variables numéricas que cambian con el tiempo y NO se conocen en el futuro
        time_varying_unknown_reals=[
            "Open", "High", "Low", "Close", "Volume"
        ],
        
        # Normalizar el target para acelerar la convergencia de la red
        target_normalizer=None, # Podemos configurar un normalizador (GroupNormalizer) más adelante
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    
    return training_dataset

if __name__ == "__main__":
    # Simulación rápida para validar la creación del tensor
    print("Este módulo está diseñado para ser importado por el orquestador principal.")
    print("Contiene la lógica para generar el TimeSeriesDataSet del TFT.")