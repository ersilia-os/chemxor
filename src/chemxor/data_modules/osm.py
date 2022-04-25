"""OSM data module."""

from pathlib import Path
from typing import Any, List, Optional

import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


class OSMDataLoader(Dataset):
    """OSM Dataset."""

    def __init__(
        self: "OSMDataLoader",
        train_set_dir: Path = Path(__file__)
        .parents[1]
        .joinpath("/data/01_raw/train_set.csv"),
        train_res_dir: Path = Path(__file__)
        .parents[1]
        .joinpath("/data/01_raw/train_res.csv"),
        transform: Optional[List[transforms]] = None,
        target_transform: Optional[List[transforms]] = None,
    ) -> None:
        """OSM dataset.

        Args:
            train_set_dir (Path): train set path. Defaults to Path(__file__).parents[1].joinpath("/data/01_raw/train_set.csv").
            train_res_dir (Path): train res path. Defaults to Path(__file__).parents[1].joinpath("/data/01_raw/train_res.csv").
        """
        self.train_set_dir = train_set_dir
        self.train_res_dir = train_res_dir
        self.train_set_df = pd.read_csv(self.train_set_df)
        self.train_res_df = pd.read_csv(self.train_res_df)
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.target_transform = target_transform

    def __len__(self: "OSMDataLoader") -> int:
        """Length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.train_res_df)

    def __getitem__(self: "OSMDataLoader", index: Any) -> Any:
        molecule = self.train_set_df[index]
        label = self.train_res_df[index]
        if self.transform:
            molecule = self.transform(molecule)
        if self.target_transform:
            label = self.target_transform(label)
        return molecule, label


class OSMDataModule(pl.LightningDataModule):
    """OSM data module."""

    def __init__(
        self: "OSMDataModule",
        train_set_dir: Path = Path(__file__)
        .parents[1]
        .joinpath("/data/01_raw/train_set.csv"),
        train_res_dir: Path = Path(__file__)
        .parents[1]
        .joinpath("/data/01_raw/train_res.csv"),
        batch_size: int = 32,
    ) -> None:
        """OSM data module init.

        Args:
            train_set_dir (Path): train set path. Defaults to Path(__file__).parents[1].joinpath("/data/01_raw/train_set.csv").
            train_res_dir (Path): train res path. Defaults to Path(__file__).parents[1].joinpath("/data/01_raw/train_res.csv").
            batch_size (int): batch size. Defaults to 32.
        """
        super().__init__()
        self.train_set_dir = train_set_dir
        self.train_res_dir = train_res_dir
        self.batch_size = batch_size

    def prepare_data(self: "OSMDataModule"):
        """Prepare data."""
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self: "OSMDataModule", stage: Optional[str]) -> None:
        """Setup dataloaders.

        Args:
            stage (Optional[str]): Optional pipeline state
        """
        osm_full = OSMDataLoader(self.train_set_dir, self.train_res_dir)
        self.osm_train, self.osm_val = random_split(osm_full, [270, 117])

    def train_dataloader(self: "OSMDataModule") -> DataLoader:
        """Train dataloader.

        Returns:
            DataLoader: train dataloader
        """
        return DataLoader(self.osm_train, batch_size=self.batch_size)

    def val_dataloader(self: "OSMDataModule"):
        """Val dataloader.

        Returns:
            DataLoader: val dataloader
        """
        return DataLoader(self.osm_val, batch_size=self.batch_size)

    def test_dataloader(self: "OSMDataModule") -> DataLoader:
        """Test data loader.

        Returns:
            DataLoader : test dataloader
        """
        return DataLoader(self.osm_val, batch_size=self.batch_size)

    def predict_dataloader(self: "OSMDataModule"):
        """Predict data loader.

        Returns:
            Dataloader : predict dataloader
        """
        return DataLoader(self.osm_predict, batch_size=self.batch_size)

    def teardown(self: "OSMDataModule"):
        """Teardown of data module."""
        pass
