"""Olinda Regression model training pipeline."""

from pathlib import Path

from kedro.pipeline import node, Pipeline
import pytorch_lightning as pl

from chemxor.data import OlindaRDataModule
from chemxor.model import OlindaNetOne
from chemxor.utils import get_project_root_path


def init_data_module() -> pl.LightningDataModule:
    """Initialize data module."""
    return OlindaRDataModule()


def init_model() -> pl.LightningModule:
    """Initialize model."""
    return OlindaNetOne(output=1)


default_path = get_project_root_path().joinpath(
    "data/06_models/olindanet-one-regression"
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
        callbacks=[checkpoint_callback], accelerator="auto", gradient_clip_val=0.5
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
olinda_one_reg_pipeline = Pipeline(
    [
        init_data_module_node,
        init_model_node,
        train_model_node,
    ]
)
