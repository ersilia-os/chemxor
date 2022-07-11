"""Smiles to Img data module."""

from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from rdkit import Chem
from rdkit.Chem import AllChem
import tenseal as ts
import torch as t
from torch.utils.data import DataLoader, Dataset, random_split

from chemxor.data_modules.enc_dataset import EncDataset
from chemxor.utils import get_project_root_path

project_root_path = get_project_root_path()
default_path = project_root_path.joinpath(
    "data/01_raw/ModelPreds/eos2r5a/ersilia_output.csv"
)


class SmilesToImgDataset(Dataset):
    """SmilesToImg Dataset."""

    def __init__(
        self: "SmilesToImgDataset",
        csv_path: Path = default_path,
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
    ) -> None:
        """Smiles to image dataset.

        Args:
            csv_path (Path): csv path.
                Defaults to default_path.
            transform (Optional[Any]): Tranforms for the inputs. Defaults to None.
            target_transform (Optional[Any]): Transforms for the target. Defaults to None.

        """
        self.df_full = pd.read_csv(csv_path.absolute())
        self.transform = transform
        self.target_transform = target_transform
        # load the transformer
        self.grid_transformer = joblib.load(
            project_root_path.joinpath("data/models/grid_transformer.joblib")
        )

    def __len__(self: "SmilesToImgDataset") -> int:
        """Length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.train_set_df)

    def __getitem__(self: "SmilesToImgDataset", index: int) -> Any:
        """Get item from the dataset.

        Args:
            index (int): index of the item.

        Returns:
            Any: Item
        """
        # Extract smile string
        molecule_smile = self.df_full.iloc[index][1]

        # Extract label
        label = t.tensor(self.df_full.iloc[index][0])

        # Create molecule figerprints
        molecule_fp = AllChem.GetMorganFingerprint(
            Chem.MolFromSmiles(molecule_smile),
            radius=3,
            useCounts=True,
            useFeatures=True,
        )

        # Truncate fp to len 1024
        molecule_fp_truncated = np.zeros((1024), np.uint8)
        for idx, v in molecule_fp.GetNonzeroElements().items():
            molecule_fp_truncated[idx] += int(v)

        # Create input image
        molecule_img = self.grid_transformer([molecule_fp_truncated])[0]

        # Apply input transforms
        if self.transform:
            molecule_img = self.transform(molecule_img)

        # Apply target transforms
        if self.target_transform:
            label = self.target_transform(label)

        # Return input and target
        return molecule_img, label


class SmilesToImgDataModule(pl.LightningDataModule):
    """OSM data module."""

    def __init__(
        self: "SmilesToImgDataset",
        csv_path: Path = default_path,
        batch_size: int = 10,
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
    ) -> None:
        """OSM data module init.

        Args:
            csv_path (Path): csv path.
                Defaults to default_path.
            batch_size (int): batch size. Defaults to 32.
            transform (Optional[Any]): Tranforms for the inputs. Defaults to None.
            target_transform (Optional[Any]): Transforms for the target. Defaults to None.
        """
        super().__init__()
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.transform = transform
        self.target_transform = target_transform

    def prepare_data(self: "SmilesToImgDataModule") -> None:
        """Prepare data."""
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self: "SmilesToImgDataModule", stage: Optional[str]) -> None:
        """Setup dataloaders.

        Args:
            stage (Optional[str]): Optional pipeline state
        """
        csv_full = SmilesToImgDataset(
            self.csv_path,
            self.transform,
            self.target_transform,
        )
        self.csv_train, self.csv_val = random_split(
            csv_full, [270, 117], t.Generator().manual_seed(7777)
        )

        # hack for now, please fixme later
        self.enc_csv_train = deepcopy(self.csv_train)
        self.csv_test = deepcopy(self.csv_val)
        self.enc_csv_test = deepcopy(self.csv_val)
        self.csv_predict = deepcopy(self.csv_val)
        self.enc_csv_predict = deepcopy(self.csv_val)

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

    def train_dataloader(self: "SmilesToImgDataModule") -> DataLoader:
        """Train dataloader.

        Returns:
            DataLoader: train dataloader
        """
        return DataLoader(self.csv_train, batch_size=self.batch_size)

    def enc_train_dataloader(
        self: "SmilesToImgDataModule", context: ts.Context
    ) -> DataLoader:
        """Encrypted train dataloader.

        Args:
            context (ts.Context): Tenseal encryption context.

        Returns:
            DataLoader: train dataloader
        """
        enc_csv_train = EncDataset(context, self.enc_csv_train)
        return DataLoader(enc_csv_train, batch_size=None)

    def val_dataloader(self: "SmilesToImgDataModule") -> DataLoader:
        """Val dataloader.

        Returns:
            DataLoader: val dataloader
        """
        return DataLoader(self.csv_val, batch_size=self.batch_size)

    def enc_val_dataloader(
        self: "SmilesToImgDataModule", context: ts.Context
    ) -> DataLoader:
        """Encrypted val dataloader.

        Args:
            context (ts.Context): Tenseal encryption context.

        Returns:
            DataLoader: val dataloader
        """
        enc_csv_val = EncDataset(context, self.enc_csv_val)
        return DataLoader(enc_csv_val, batch_size=None)

    def test_dataloader(self: "SmilesToImgDataModule") -> DataLoader:
        """Test data loader.

        Returns:
            DataLoader : test dataloader
        """
        return DataLoader(self.csv_test, batch_size=self.batch_size)

    def enc_test_dataloader(
        self: "SmilesToImgDataModule", context: ts.Context
    ) -> DataLoader:
        """Encrypted test dataloader.

        Args:
            context (ts.Context): Tenseal encryption context.

        Returns:
            DataLoader: test dataloader
        """
        enc_csv_test = EncDataset(context, self.enc_csv_test)
        return DataLoader(enc_csv_test, batch_size=None)

    def predict_dataloader(self: "SmilesToImgDataModule") -> DataLoader:
        """Predict data loader.

        Returns:
            DataLoader : predict dataloader
        """
        return DataLoader(self.csv_predict, batch_size=self.batch_size)

    def enc_predict_dataloader(
        self: "SmilesToImgDataModule", context: ts.Context
    ) -> DataLoader:
        """Encrypted predict dataloader.

        Args:
            context (ts.Context): Tenseal encryption context.

        Returns:
            DataLoader: predict dataloader
        """
        enc_csv_predict = EncDataset(context, self.enc_csv_predict)
        return DataLoader(enc_csv_predict, batch_size=None)

    def teardown(self: "SmilesToImgDataModule", stage: Optional[str] = None) -> None:
        """Teardown of data module."""
        pass
