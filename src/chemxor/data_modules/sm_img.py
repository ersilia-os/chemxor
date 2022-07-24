"""SmilesImg data module."""

from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Union

import dask.dataframe as dd
import pandas as pd
import pytorch_lightning as pl
import tenseal as ts
import torch as t
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from chemxor.data_modules.enc_conv_dataset import EncConvDataset
from chemxor.utils import get_project_root_path

project_root_path = get_project_root_path()
default_path = project_root_path.joinpath(
    "data/01_raw/ModelPreds/eos2r5a/ersilia_output.csv"
)


class SmilesImgDataset(Dataset):
    """SmilesImg Dataset."""

    def __init__(
        self: "SmilesImgDataset",
        preds_csv_path: Path = default_path,
        imgs_df_dir: Path = default_path.parents[0],
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
    ) -> None:
        """Smiles as image dataset.

        Args:
            preds_csv_path (Path): preds csv path.
                Defaults to default_path.
            imgs_df_dir (Path): images csv path.
                Defaults to default_path.parents[0].
            transform (Optional[Any]): Tranforms for the inputs. Defaults to None.
            target_transform (Optional[Any]): Transforms for the target. Defaults to None.

        """
        self.preds_df_full = pd.read_csv(preds_csv_path.absolute())
        self.imgs_df_full = dd.read_csv(imgs_df_dir.joinpath("sm_to_imgs_*"))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self: "SmilesImgDataset") -> int:
        """Length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.preds_df_full)

    def __getitem__(self: "SmilesImgDataset", index: int) -> Any:
        """Get item from the dataset.

        Args:
            index (int): index of the item.

        Returns:
            Any: Item
        """
        # Extract smile string
        molecule_img = t.tensor(self.imgs_df_full.iloc[index], dtype=t.float).reshape(
            1, 32, 32
        )

        # Extract target
        label = t.tensor(self.preds_df_full.iloc[index][0], dtype=t.float)

        # Apply input transforms
        if self.transform:
            molecule_img = self.transform(molecule_img)

        # Apply target transforms
        if self.target_transform:
            label = self.target_transform(label)

        # Return input and target
        return molecule_img, label


class SmilesImgDataModule(pl.LightningDataModule):
    """SmilesImg data module."""

    def __init__(
        self: "SmilesImgDataset",
        csv_path: Path = default_path,
        imgs_df_dir: Path = default_path.parents[0],
        batch_size: int = 32,
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
        model: Optional[Union[nn.Module, pl.LightningModule]] = None,
    ) -> None:
        """OSM data module init.

        Args:
            csv_path (Path): csv path.
                Defaults to default_path.
            imgs_df_dir (Path): images csv path.
                Defaults to default_path.parents[0].
            batch_size (int): batch size. Defaults to 32.
            transform (Optional[Any]): Tranforms for the inputs. Defaults to None.
            target_transform (Optional[Any]): Transforms for the target. Defaults to None.
            model (Optional[Union[nn.Module, pl.LightningModule]]): Model. Default to None.
        """
        super().__init__()
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.transform = transform
        self.target_transform = target_transform
        self.model = model

    def prepare_data(self: "SmilesImgDataModule") -> None:
        """Prepare data."""
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self: "SmilesImgDataModule", stage: Optional[str]) -> None:
        """Setup dataloaders.

        Args:
            stage (Optional[str]): Optional pipeline state
        """
        csv_full = SmilesImgDataset(
            self.csv_path,
            self.transform,
            self.target_transform,
        )

        train_len = int(len(csv_full) * 0.8)
        validate_len = int(len(csv_full) * 0.1)
        test_len = int(len(csv_full) * 0.1)
        others_len = len(csv_full) - train_len - validate_len - test_len
        self.csv_train, self.csv_val, self.csv_test, _ = random_split(
            csv_full,
            [train_len, validate_len, test_len, others_len],
        )

        # hack for now, please fixme later
        self.enc_csv_train = deepcopy(self.csv_train)
        self.enc_csv_test = deepcopy(self.csv_test)
        self.csv_predict = deepcopy(self.csv_test)
        self.enc_csv_predict = deepcopy(self.csv_test)

    def train_dataloader(self: "SmilesImgDataModule") -> DataLoader:
        """Train dataloader.

        Returns:
            DataLoader: train dataloader
        """
        return DataLoader(self.csv_train, batch_size=self.batch_size)

    def enc_train_dataloader(
        self: "SmilesImgDataModule", context: ts.Context
    ) -> DataLoader:
        """Encrypted train dataloader.

        Args:
            context (ts.Context): Tenseal encryption context.

        Returns:
            DataLoader: train dataloader
        """
        enc_csv_train = EncConvDataset(
            context,
            self.enc_csv_train,
            self.model.conv1.kernel_size,
            self.model.conv1.stride[0],
            (32, 32),
        )
        return DataLoader(enc_csv_train, batch_size=None)

    def val_dataloader(self: "SmilesImgDataModule") -> DataLoader:
        """Val dataloader.

        Returns:
            DataLoader: val dataloader
        """
        return DataLoader(self.csv_val, batch_size=self.batch_size)

    def enc_val_dataloader(
        self: "SmilesImgDataModule", context: ts.Context
    ) -> DataLoader:
        """Encrypted val dataloader.

        Args:
            context (ts.Context): Tenseal encryption context.

        Returns:
            DataLoader: val dataloader
        """
        enc_csv_val = EncConvDataset(
            context,
            self.enc_csv_val,
            self.model.conv1.kernel_size,
            self.model.conv1.stride[0],
            (32, 32),
        )
        return DataLoader(enc_csv_val, batch_size=None)

    def test_dataloader(self: "SmilesImgDataModule") -> DataLoader:
        """Test data loader.

        Returns:
            DataLoader : test dataloader
        """
        return DataLoader(self.csv_test, batch_size=self.batch_size)

    def enc_test_dataloader(
        self: "SmilesImgDataModule", context: ts.Context
    ) -> DataLoader:
        """Encrypted test dataloader.

        Args:
            context (ts.Context): Tenseal encryption context.

        Returns:
            DataLoader: test dataloader
        """
        enc_csv_test = EncConvDataset(
            context,
            self.enc_csv_test,
            self.model.conv1.kernel_size,
            self.model.conv1.stride[0],
            (32, 32),
        )
        return DataLoader(enc_csv_test, batch_size=None)

    def predict_dataloader(self: "SmilesImgDataModule") -> DataLoader:
        """Predict data loader.

        Returns:
            DataLoader : predict dataloader
        """
        return DataLoader(self.csv_predict, batch_size=self.batch_size)

    def enc_predict_dataloader(
        self: "SmilesImgDataModule", context: ts.Context
    ) -> DataLoader:
        """Encrypted predict dataloader.

        Args:
            context (ts.Context): Tenseal encryption context.

        Returns:
            DataLoader: predict dataloader
        """
        enc_csv_predict = EncConvDataset(
            context,
            self.enc_csv_predict,
            self.model.conv1.kernel_size,
            self.model.conv1.stride[0],
            (32, 32),
        )
        return DataLoader(enc_csv_predict, batch_size=None)

    def teardown(self: "SmilesImgDataModule", stage: Optional[str] = None) -> None:
        """Teardown of data module."""
        pass
