"Cryptic Sage default nodes."

from kedro.pipeline import node
import pytorch_lightning as pl

from chemxor.data_modules.osm import OSMDataModule
from chemxor.model.cryptic_sage import CrypticSage


def train_model() -> None:
    """Train model node funciton."""
    osm = OSMDataModule()
    model = CrypticSage()
    trainer = pl.Trainer(limit_train_batches=100, max_epochs=1)
    trainer.fit(model=model, datamodule=osm)


train_model_node = node(func=train_model, inputs=None, outputs=None)
