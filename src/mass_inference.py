import os
import glob
import warnings
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from src.tft_dataset import add_time_features

warnings.filterwarnings("ignore")

def calculate_mdd(cumulative_returns):
    peaks = cumulative_returns.cummax()
    drawdowns = (cumulative_returns - peaks) / peaks
    return drawdowns.min()

def run_mass_inference():
    print("=== Iniciando Motor de Inferencia Masiva (Transfer Learning) ===")
    
    best_model_path = "models/best/best_model.ckpt"
    print(f"Cargando modelo: {best_model_path}\n")
    model = TemporalFusionTransformer.load_from_checkpoint(best_model_path, weights_only=False)
    model.eval()

    # 2. Leer los 500+ ETFs de la carpeta raw
    raw_files = glob.glob("data/raw/*.csv")
    print(f"Se encontraron {len(raw_files)} ETFs para validación masiva.")
    
    results = []
    
    for file_path in raw_files:
        ticker = os.path.basename(file_path).replace(".csv", "")
        print(f"Analizando: {ticker}...", end=" ")
        
        try:
            df = pd.read_csv(file_path)
            if 'Date' not in df.columns or len(df) < 100:
                print("Saltado (Datos insuficientes o formato incorrecto).")
                continue
                
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
            
            # TRANSFER LEARNING HACK: Mapeamos el ID categórico a uno conocido 
            # para que el modelo procese la serie temporal sin error de "Unseen Label"
            df['Symbol'] = "VOO" 
            
            df = add_time_features(df)
            
            # Recreamos el dataset usando los parámetros exactos del modelo entrenado
            dataset = TimeSeriesDataSet.from_parameters(
                model.dataset_parameters, df, predict=True, stop_randomization=True
            )
            dataloader = dataset.to_dataloader(train=False, batch_size=128, num_workers=0)
            
            # Inferencia
            predictions = model.predict(dataloader, mode="prediction", return_y=True)
            
            # Extraer mediana de cuantiles (índice 3) o valor directo
            if len(predictions.output.shape) == 3:
                y_pred = predictions.output[:, :, 3].numpy().flatten()
            else:
                y_pred = predictions.output.numpy().flatten()
                
            y_true = predictions.y[0].numpy().flatten()
            
            # Lógica de Trading (Direccionalidad)
            diff_pred = np.diff(y_pred)
            diff_true = np.diff(y_true)
            
            signals_pred = (diff_pred > 0).astype(int)
            signals_true = (diff_true > 0).astype(int)
            
            # F1-Score
            f1 = f1_score(signals_true, signals_pred, average='weighted', zero_division=0)
            
            # Cálculo de Drawdowns (Riesgo)
            # Evitamos divisiones por cero sumando un epsilon minúsculo
            actual_returns = diff_true / (y_true[:-1] + 1e-8)
            
            cumulative_market = pd.Series((1 + actual_returns).cumprod())
            cumulative_agent = pd.Series((1 + (actual_returns * signals_pred)).cumprod())
            
            mdd_market = calculate_mdd(cumulative_market) * 100
            mdd_agent = calculate_mdd(cumulative_agent) * 100
            
            # Registrar resultado
            results.append({
                "Ticker": ticker,
                "F1_Score": round(f1, 4),
                "MDD_Mercado_%": round(mdd_market, 2),
                "MDD_Agente_%": round(mdd_agent, 2),
                "Riesgo_Reducido": "Sí" if mdd_agent > mdd_market else "No" # > porque MDD es negativo
            })
            print(f"F1: {f1:.2f} | MDD Mercado: {mdd_market:.1f}% | MDD Agente: {mdd_agent:.1f}%")
            
        except Exception as e:
            print(f"Error procesando {ticker}: {str(e)}")
            
    # 3. Guardar el reporte maestro
    if results:
        results_df = pd.DataFrame(results)
        output_path = "data/processed/mass_inference_results.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results_df.to_csv(output_path, index=False)
        
        # Resumen global
        exitos = len(results_df[results_df['Riesgo_Reducido'] == 'Sí'])
        print("\n" + "="*50)
        print("🏁 INFERENCIA MASIVA COMPLETADA")
        print("="*50)
        print(f"Total ETFs analizados   : {len(results)}")
        print(f"ETFs protegidos por TFT : {exitos} ({(exitos/len(results))*100:.1f}%)")
        print(f"MDD Promedio (Mercado)  : {results_df['MDD_Mercado_%'].mean():.2f}%")
        print(f"MDD Promedio (Agente)   : {results_df['MDD_Agente_%'].mean():.2f}%")
        print(f"\nReporte detallado guardado en: {output_path}")

if __name__ == "__main__":
    run_mass_inference()