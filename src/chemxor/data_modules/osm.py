"""OSM data module."""

from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pytorch_lightning as pl
import tenseal as ts
import torch as t
from torch.utils.data import DataLoader, Dataset, random_split

from chemxor.data_modules.enc_dataset import EncDataset


class OSMDataset(Dataset):
    """OSM Dataset."""

    def __init__(
        self: "OSMDataset",
        train_set_dir_path: Path = (
            Path(__file__)  # noqa: B008
            .parents[3]
            .joinpath("data/01_raw/train_set.csv")
        ),
        train_res_dir_path: Path = (
            Path(__file__)  # noqa: B008
            .parents[3]
            .joinpath("data/01_raw/train_res.csv")
        ),
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
    ) -> None:
        """OSM dataset.

        Args:
            train_set_dir_path (Path): train set path.
                Defaults to Path(__file__).parents[1].joinpath("/data/01_raw/train_set.csv").
            train_res_dir_path (Path): train res path.
                Defaults to Path(__file__).parents[1].joinpath("/data/01_raw/train_res.csv").
            transform (Optional[Any]): Tranforms for the inputs. Defaults to None.
            target_transform (Optional[Any]): Transforms for the target. Defaults to None.

        """
        self.train_set_df = pd.read_csv(train_set_dir_path.absolute())
        self.train_res_df = pd.read_csv(train_res_dir_path.absolute())
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self: "OSMDataset") -> int:
        """Length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.train_set_df)

    def __getitem__(self: "OSMDataset", index: int) -> Any:
        """Get item from the dataset.

        Args:
            index (int): index of the item.

        Returns:
            Any: Item
        """
        molecule = t.tensor(self.train_set_df.iloc[index][2:], dtype=t.float32)
        label = self.train_res_df.iloc[index][2]
        if label < 1:
            label = t.tensor([1, 0], dtype=t.float32)
        else:
            label = t.tensor([0, 1], dtype=t.float32)
        if self.transform:
            molecule = self.transform(molecule)
        if self.target_transform:
            label = self.target_transform(label)
        return molecule, label


class OSMDataModule(pl.LightningDataModule):
    """OSM data module."""

    def __init__(
        self: "OSMDataModule",
        train_set_dir_path: Path = (
            Path(__file__)  # noqa: B008
            .parents[3]
            .joinpath("data/01_raw/train_set.csv")
        ),
        train_res_dir_path: Path = (
            Path(__file__)  # noqa: B008
            .parents[3]
            .joinpath("data/01_raw/train_res.csv")
        ),
        batch_size: int = 10,
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
    ) -> None:
        """OSM data module init.

        Args:
            train_set_dir_path (Path): train set path.
                Defaults to Path(__file__).parents[1].joinpath("/data/01_raw/train_set.csv").
            train_res_dir_path (Path): train res path.
                Defaults to Path(__file__).parents[1].joinpath("/data/01_raw/train_res.csv").
            batch_size (int): batch size. Defaults to 32.
            transform (Optional[Any]): Tranforms for the inputs. Defaults to None.
            target_transform (Optional[Any]): Transforms for the target. Defaults to None.
        """
        super().__init__()
        self.train_set_dir_path = train_set_dir_path
        self.train_res_dir_path = train_res_dir_path
        self.batch_size = batch_size
        self.transform = transform
        self.target_transform = target_transform

    def prepare_data(self: "OSMDataModule") -> None:
        """Prepare data."""
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self: "OSMDataModule", stage: Optional[str]) -> None:
        """Setup dataloaders.

        Args:
            stage (Optional[str]): Optional pipeline state
        """
        osm_full = OSMDataset(
            self.train_set_dir_path,
            self.train_res_dir_path,
            self.transform,
            self.target_transform,
        )
        self.osm_train, self.osm_val = random_split(
            osm_full, [270, 117], t.Generator().manual_seed(7777)
        )

        # hack for now, please fixme later
        self.enc_osm_train = deepcopy(self.osm_train)
        self.osm_test = deepcopy(self.osm_val)
        self.enc_osm_test = deepcopy(self.osm_val)
        self.osm_predict = deepcopy(self.osm_val)
        self.enc_osm_predict = deepcopy(self.osm_val)

        # # split train into train and validation
        # self.osm_train, self.osm_val = random_split(
        #     self.osm_train,
        #     [
        #         int(len(self.osm_train) * 80),
        #         len(self.osm_train) - int(len(self.osm_train) * 80),
        #     ],
        # )
        # # split test into test and predict
        # self.osm_test, self.osm_predict = random_split(
        #     self.osm_test,
        #     [
        #         int(len(self.osm_test) * 80),
        #         len(self.osm_test) - int(len(self.osm_test) * 80),
        #     ],
        # )

    def train_dataloader(self: "OSMDataModule") -> DataLoader:
        """Train dataloader.

        Returns:
            DataLoader: train dataloader
        """
        return DataLoader(self.osm_train, batch_size=self.batch_size)

    def enc_train_dataloader(self: "OSMDataModule", context: ts.Context) -> DataLoader:
        """Encrypted train dataloader.

        Args:
            context (ts.Context): Tenseal encryption context.

        Returns:
            DataLoader: train dataloader
        """
        enc_osm_train = EncDataset(context, self.enc_osm_train)
        return DataLoader(enc_osm_train, batch_size=self.batch_size)

    def val_dataloader(self: "OSMDataModule") -> DataLoader:
        """Val dataloader.

        Returns:
            DataLoader: val dataloader
        """
        return DataLoader(self.osm_val, batch_size=self.batch_size)

    def enc_val_dataloader(self: "OSMDataModule", context: ts.Context) -> DataLoader:
        """Encrypted val dataloader.

        Args:
            context (ts.Context): Tenseal encryption context.

        Returns:
            DataLoader: val dataloader
        """
        enc_osm_val = EncDataset(context, self.enc_osm_val)
        return DataLoader(enc_osm_val, batch_size=self.batch_size)

    def test_dataloader(self: "OSMDataModule") -> DataLoader:
        """Test data loader.

        Returns:
            DataLoader : test dataloader
        """
        return DataLoader(self.osm_test, batch_size=self.batch_size)

    def enc_test_dataloader(self: "OSMDataModule", context: ts.Context) -> DataLoader:
        """Encrypted test dataloader.

        Args:
            context (ts.Context): Tenseal encryption context.

        Returns:
            DataLoader: test dataloader
        """
        enc_osm_test = EncDataset(context, self.enc_osm_test)
        return DataLoader(enc_osm_test, batch_size=self.batch_size)

    def predict_dataloader(self: "OSMDataModule") -> DataLoader:
        """Predict data loader.

        Returns:
            DataLoader : predict dataloader
        """
        return DataLoader(self.osm_predict, batch_size=self.batch_size)

    def enc_predict_dataloader(
        self: "OSMDataModule", context: ts.Context
    ) -> DataLoader:
        """Encrypted predict dataloader.

        Args:
            context (ts.Context): Tenseal encryption context.

        Returns:
            DataLoader: predict dataloader
        """
        enc_osm_predict = EncDataset(context, self.enc_osm_predict)
        return DataLoader(enc_osm_predict, batch_size=self.batch_size)

    def teardown(self: "OSMDataModule", stage: Optional[str] = None) -> None:
        """Teardown of data module."""
        pass
