import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss

def train_tft_model(training_dataset: TimeSeriesDataSet, val_dataset: TimeSeriesDataSet, batch_size=64):
    """
    Entrena el modelo Temporal Fusion Transformer utilizando PyTorch Lightning
    y registra el experimento de forma nativa en MLflow.
    """
    print("Preparando los DataLoaders para entrenamiento y validación...")
    
    train_dataloader = training_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = val_dataset.to_dataloader(train=False, batch_size=batch_size * 2, num_workers=0)

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=5,
        verbose=True,
        mode="min"
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="models/",
        filename="tft-best-checkpoint",
        save_top_k=1,
        mode="min"
    )

    mlf_logger = MLFlowLogger(
        experiment_name="Agente_Restrictivo_Preentrenamiento",
        tracking_uri="file:./mlruns"
    )

    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="auto", 
        callbacks=[early_stop_callback, checkpoint_callback],
        enable_progress_bar=True,
        logger=mlf_logger
    )

    tft_model = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=8,
        output_size=7,
        loss=QuantileLoss(),
        reduce_on_plateau_patience=4
    )

    print(f"Número de parámetros en la red: {tft_model.size()/1e3:.1f}k")
    print("Iniciando el entrenamiento algorítmico...")
    
    trainer.fit(
        tft_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )
        
    print("Entrenamiento completado y registrado en MLflow.")
        
    return tft_model, trainer

if __name__ == "__main__":
    print("Módulo de entrenamiento.")