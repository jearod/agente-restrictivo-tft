import pandas as pd
import os

# Corpus Global Combinado: Mantenemos los originales y completamos con la metadata disponible
# América (Pilares y capitalizaciones medias/bajas)
# Europa (Exposición a mercados desarrollados europeos)
# Asia / Pacífico (Exposición a Japón, China, India y la región del Pacífico)
GLOBAL_ETFS = [
    "VOO", "QQQ", "DIA", "IJH", "IJR",
    "IEUS", "EWL", "EWD",
    "EWJV", "VPL", "CNYA", "SMIN"
]

def load_and_sort_data(filepath):
    df = pd.read_csv(filepath)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
    return df

def strict_temporal_partition(df):
    total_rows = len(df)
    train_end = int(total_rows * 0.70)
    val_end = int(total_rows * 0.85)
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    return train_df, val_df, test_df

def load_global_portfolio(data_dir="data/pretrain"):
    all_train, all_val, all_test = [], [], []
    
    print(f"Buscando el corpus global de {len(GLOBAL_ETFS)} ETFs en '{data_dir}'...")
    
    for ticker in GLOBAL_ETFS:
        filepath = os.path.join(data_dir, f"{ticker}.csv")
        
        if os.path.exists(filepath):
            print(f" - Procesando: {ticker}")
            df = load_and_sort_data(filepath)
            df['Symbol'] = ticker 
            
            # Particionamos CADA ETF individualmente
            train, val, test = strict_temporal_partition(df)
            
            all_train.append(train)
            all_val.append(val)
            all_test.append(test)
        else:
            print(f" [!] Advertencia: Archivo {ticker}.csv no encontrado. Saltando...")
            
    if not all_train:
        raise FileNotFoundError("No se encontró ningún archivo CSV de los ETFs especificados.")
        
    # Concatenamos todo en tensores masivos
    mega_train = pd.concat(all_train, ignore_index=True)
    mega_val = pd.concat(all_val, ignore_index=True)
    mega_test = pd.concat(all_test, ignore_index=True)
    
    print(f"\nCorpus consolidado:")
    print(f" - Entrenamiento Total: {len(mega_train)} registros")
    
    return mega_train, mega_val, mega_test