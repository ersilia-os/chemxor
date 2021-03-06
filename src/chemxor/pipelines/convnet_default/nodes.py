"Convnet default nodes."

from kedro.pipeline import node
import pytorch_lightning as pl

from chemxor.data_modules.mnist import MNISTDataModule
from chemxor.model.convnet import ConvNet
from chemxor.utils import get_project_root_path


def init_mnist_data_module() -> pl.LightningDataModule:
    """Initialize mnist data module."""
    return MNISTDataModule()


def init_convnet_model() -> pl.LightningModule:
    """Initialize Convnet model."""
    return ConvNet()


def train_convnet_model(
    data_module: pl.LightningDataModule,
    model: pl.LightningDataModule,
) -> None:
    """Train model."""
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=get_project_root_path().joinpath("data/06_models/convnet"),
        save_top_k=3,
        monitor="validation_loss",
    )
    trainer = pl.Trainer(callbacks=[checkpoint_callback], accelerator="auto")
    trainer.fit(model=model, datamodule=data_module)


init_mnist_data_module_node = node(
    func=init_mnist_data_module, inputs=None, outputs="mnist_data_module"
)
init_convnet_model_node = node(
    func=init_convnet_model, inputs=None, outputs="convnet_model"
)
train_convnet_model_node = node(
    func=train_convnet_model,
    inputs=["mnist_data_module", "convnet_model"],
    outputs=None,
)
