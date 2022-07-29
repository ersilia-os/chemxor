"""Nodes."""

from kedro.pipeline import node
import pytorch_lightning as pl
from torch.nn import functional as F

from chemxor.data_modules.sm_img import SmilesImgDataModule
from chemxor.model.blocks import ConvLinearOne
from chemxor.utils import get_project_root_path


def init_sm_img_data_module() -> pl.LightningDataModule:
    """Initialize smiles img data module."""
    return SmilesImgDataModule()


def init_convnet_model() -> pl.LightningModule:
    """Initialize Convnet model."""
    return ConvLinearOne(output=1, criterion=F.mse_loss)


def train_convnet_model(
    data_module: pl.LightningDataModule,
    model: pl.LightningDataModule,
) -> None:
    """Train model."""
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=get_project_root_path().joinpath("data/06_models/convnet-linear-one"),
        save_top_k=3,
        monitor="VAL_Loss",
    )
    trainer = pl.Trainer(callbacks=[checkpoint_callback], accelerator="auto")
    trainer.fit(model=model, datamodule=data_module)


init_smiles_data_module_node = node(
    func=init_sm_img_data_module, inputs=None, outputs="smiles_data_module"
)
init_convnet_model_node = node(
    func=init_convnet_model, inputs=None, outputs="convnet_model"
)
train_convnet_model_node = node(
    func=train_convnet_model,
    inputs=["smiles_data_module", "convnet_model"],
    outputs=None,
)
