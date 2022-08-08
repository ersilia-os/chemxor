"""Olinda Classification model training pipeline."""

from pathlib import Path

from kedro.pipeline import node, Pipeline
import pytorch_lightning as pl

from chemxor.data import OlindaCDataModule
from chemxor.model import OlindaNetOne
from chemxor.utils import get_project_root_path


def init_data_module() -> pl.LightningDataModule:
    """Initialize data module."""
    return OlindaCDataModule()


def init_model() -> pl.LightningModule:
    """Initialize model."""
    return OlindaNetOne(output=2)


default_path = get_project_root_path().joinpath(
    "data/06_models/olindanet-one-classification"
)


def train_model(
    data_module: pl.LightningDataModule,
    model: pl.LightningDataModule,
    checkpoint_path: Path = default_path,
) -> None:
    """Train model."""
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_path,
        save_top_k=3,
        monitor="VAL_Loss",
    )
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        accelerator="auto",
        gradient_clip_val=0.5,
        val_check_interval=0.10,
    )
    trainer.fit(model=model, datamodule=data_module)


init_data_module_node = node(func=init_data_module, inputs=None, outputs="data_module")
init_model_node = node(func=init_model, inputs=None, outputs="model")
train_model_node = node(
    func=train_model,
    inputs=["data_module", "model"],
    outputs=None,
)

# Assemble nodes into a pipeline
olinda_one_cls_pipeline = Pipeline(
    [
        init_data_module_node,
        init_model_node,
        train_model_node,
    ]
)
