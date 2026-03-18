import os
import glob
import torch
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import f1_score, classification_report
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

from src.data_pipeline import load_global_portfolio
from src.tft_dataset import add_time_features, create_tft_dataset

# Ocultar advertencias futuras de Pandas y PyTorch para una consola limpia
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def calculate_mdd(cumulative_returns):
    peaks = cumulative_returns.cummax()
    drawdowns = (cumulative_returns - peaks) / peaks
    return drawdowns.min()

def evaluate_agent():
    print("=== Iniciando Evaluación del Agente Restrictivo ===")
    
    checkpoints = glob.glob("models/*.ckpt")
    if not checkpoints:
        print("[ERROR] No se encontraron modelos en 'models/'.")
        return
    
    best_model_path = "models/best/best_model.ckpt"
    print(f"\nCargando pesos neuronales desde: {best_model_path}")
    model = TemporalFusionTransformer.load_from_checkpoint(best_model_path, weights_only=False)
    model.eval()

    print("\nExtrayendo el 15% de datos de prueba (Test Set) no vistos...")
    train_df, _, test_df = load_global_portfolio("data/pretrain")
    
    # Preprocesamiento sin el FutureWarning
    train_df = train_df.groupby('Symbol', group_keys=False).apply(lambda x: add_time_features(x))
    test_df = test_df.groupby('Symbol', group_keys=False).apply(lambda x: add_time_features(x))
    
    training_dataset = create_tft_dataset(train_df)
    test_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, test_df, predict=True, stop_randomization=True
    )
    
    test_dataloader = test_dataset.to_dataloader(train=False, batch_size=64, num_workers=0)

    print("\nRealizando predicciones tensoriales (Inferencia)...")
    predictions_output = model.predict(test_dataloader, mode="prediction", return_x=True, return_y=True)
    
    # Extraer la mediana de la predicción (si el modelo retorna cuantiles) o el valor directo
    if len(predictions_output.output.shape) == 3:
        y_pred = predictions_output.output[:, :, 3].numpy().flatten() # Índice 3 suele ser la mediana en quantiles
    else:
        y_pred = predictions_output.output.numpy().flatten()
        
    y_true = predictions_output.y[0].numpy().flatten()

    # ---------------------------------------------------------
    # 4. LÓGICA REAL DEL AGENTE RESTRICTIVO (Direccionalidad Diaria)
    # ---------------------------------------------------------
    # Calculamos el cambio de precio de un día para el siguiente: Precio(t) - Precio(t-1)
    diff_pred = np.diff(y_pred)
    diff_true = np.diff(y_true)
    
    # Señal: 1 si predecimos que sube (Óptimo), 0 si predecimos que baja o se mantiene (Subóptimo)
    signals_pred = (diff_pred > 0).astype(int)
    signals_true = (diff_true > 0).astype(int)

    # --- MÉTRICA 1: F1-Score ---
    f1 = f1_score(signals_true, signals_pred, average='weighted')
    
    print("\n" + "="*40)
    print("📊 REPORTE DE RENDIMIENTO REAL (F1-SCORE DIRECCIONAL)")
    print("="*40)
    print(classification_report(signals_true, signals_pred, target_names=["Día Subóptimo (0)", "Día Óptimo (1)"]))
    print(f"F1-Score Global: {f1:.4f}")

    # --- MÉTRICA 2: Maximum Drawdown (MDD) ---
    # Usamos los retornos reales direccionales de los precios verdaderos
    # Para evitar división por cero en retornos, usamos una aproximación logarítmica o simple
    actual_returns = diff_true / y_true[:-1] 
    
    # El mercado siempre invierte (Buy & Hold)
    market_strategy = actual_returns
    # El Agente solo invierte si su señal predictiva fue 1
    agent_strategy = actual_returns * signals_pred

    cumulative_market = pd.Series((1 + market_strategy).cumprod())
    cumulative_agent = pd.Series((1 + agent_strategy).cumprod())
    
    mdd_market = calculate_mdd(cumulative_market)
    mdd_agent = calculate_mdd(cumulative_agent)

    print("\n" + "="*40)
    print("📉 REPORTE DE RIESGO (MAXIMUM DRAWDOWN)")
    print("="*40)
    print(f"MDD Mercado (Buy & Hold) : {mdd_market*100:.2f}%")
    print(f"MDD Agente Restrictivo   : {mdd_agent*100:.2f}%")
    
    if mdd_agent > mdd_market:
        print("\n✅ ÉXITO: El Agente Restrictivo protegió el capital reduciendo la peor caída.")
    else:
        print("\n⚠️ AVISO: El Agente necesita más refinamiento para superar al mercado en riesgo.")

if __name__ == "__main__":
    evaluate_agent()