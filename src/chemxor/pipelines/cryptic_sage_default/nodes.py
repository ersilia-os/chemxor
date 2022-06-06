"Cryptic Sage default nodes."

from kedro.pipeline import node
import pytorch_lightning as pl

from chemxor.data_modules.osm import OSMDataModule
from chemxor.model.cryptic_sage import CrypticSage
from chemxor.utils import get_project_root_path


def init_osm_data_module() -> pl.LightningDataModule:
    """Initialize OSM data module."""
    return OSMDataModule()


def init_cryptic_sage_model() -> pl.LightningModule:
    """Initialize Cryptic Sage model."""
    return CrypticSage()


def train_cryptic_sage_model(
    osm_data_module: pl.LightningDataModule,
    cryptic_sage_model: pl.LightningDataModule,
) -> None:
    """Train model."""
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=get_project_root_path().joinpath("data/06_models"),
        save_top_k=3,
        monitor="validation_loss",
    )
    trainer = pl.Trainer(callbacks=[checkpoint_callback])
    trainer.fit(model=cryptic_sage_model, datamodule=osm_data_module)


init_osm_data_module_node = node(
    func=init_osm_data_module, inputs=None, outputs="osm_data_module"
)
init_cryptic_sage_model_node = node(
    func=init_cryptic_sage_model, inputs=None, outputs="cryptic_sage_model"
)
train_cryptic_sage_model_node = node(
    func=train_cryptic_sage_model,
    inputs=["osm_data_module", "cryptic_sage_model"],
    outputs=None,
)
