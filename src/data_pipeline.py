import pandas as pd
import os

def load_and_sort_data(filepath):
    """
    Carga el dataset del ETF y lo ordena cronológicamente.
    Esto es crucial para procesar series temporales.
    """
    print(f"Cargando datos desde: {filepath}")
    df = pd.read_csv(filepath)
    
    # Asegurarnos de que la columna de fecha sea un objeto 'datetime'
    # Ajusta 'Date' si tu columna tiene otro nombre (ej. 'fecha')
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        # Ordenar de más antiguo a más reciente
        df = df.sort_values('Date').reset_index(drop=True)
    
    return df

def strict_temporal_partition(df):
    """
    Aplica el Protocolo Estricto de Partición Temporal (70% / 15% / 15%).
    Ejecuta una división cronológica determinista para evitar la 
    contaminación del modelo (sesgo de anticipación).
    """
    total_rows = len(df)
    
    # Calcular los índices de corte
    train_end = int(total_rows * 0.70)
    val_end = int(total_rows * 0.85) # 70% + 15%
    
    # División secuencial aislada
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    print(f"Partición completada:")
    print(f" - Entrenamiento (70%): {len(train_df)} registros")
    print(f" - Validación (15%): {len(val_df)} registros")
    print(f" - Prueba (15%): {len(test_df)} registros")
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    # Instrucciones de prueba local:
    # 1. Coloca un archivo de prueba llamado 'etf_sample.csv' en 'data/raw/'
    # 2. Ejecuta este script
    
    sample_path = "data/raw/etf_sample.csv"
    
    # Solo ejecutamos si el archivo de prueba existe
    if os.path.exists(sample_path):
        data = load_and_sort_data(sample_path)
        train, val, test = strict_temporal_partition(data)
        
        # Opcional: Guardar los datos procesados para el entrenamiento
        # train.to_csv("../data/processed/train.csv", index=False)
        # val.to_csv("../data/processed/val.csv", index=False)
        # test.to_csv("../data/processed/test.csv", index=False)
    else:
        print(f"Para probar el script, por favor añade un archivo CSV en {sample_path}")