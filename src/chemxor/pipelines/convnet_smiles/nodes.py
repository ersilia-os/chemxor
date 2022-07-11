"Convnet default nodes."

from kedro.pipeline import node
import pytorch_lightning as pl

from chemxor.data_modules.smiles_to_img import SmilesToImgDataModule
from chemxor.model.convnet import ConvNet
from chemxor.utils import get_project_root_path


def init_smiles_data_module() -> pl.LightningDataModule:
    """Initialize smiles data module."""
    return SmilesToImgDataModule()


def init_convnet_model() -> pl.LightningModule:
    """Initialize Convnet model."""
    return ConvNet()


def train_convnet_model(
    data_module: pl.LightningDataModule,
    model: pl.LightningDataModule,
) -> None:
    """Train model."""
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=get_project_root_path().joinpath("data/06_models/convnet-smiles"),
        save_top_k=3,
        monitor="validation_loss",
    )
    trainer = pl.Trainer(callbacks=[checkpoint_callback], accelerator="auto")
    trainer.fit(model=model, datamodule=data_module)


init_smiles_data_module_node = node(
    func=init_smiles_data_module, inputs=None, outputs="smiles_data_module"
)
init_convnet_model_node = node(
    func=init_convnet_model, inputs=None, outputs="convnet_model"
)
train_convnet_model_node = node(
    func=train_convnet_model,
    inputs=["smiles_data_module", "convnet_model"],
    outputs=None,
)
