"""MNIST data module."""

from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Optional, Union

import pytorch_lightning as pl
import tenseal as ts
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from chemxor.data.enc_conv_dataset import EncConvDataset
from chemxor.utils import get_package_root_path

package_root_path = get_package_root_path()


# Adapted from https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html#datamodules
class MNISTDataModule(pl.LightningDataModule):
    """MNIST data module."""

    def __init__(
        self: "MNISTDataModule",
        data_dir: Path = package_root_path,
        batch_size: int = 32,
        transform: Optional[Any] = transforms.Compose(  # noqa: B008
            [
                transforms.ToTensor(),  # noqa: B008
                transforms.Normalize((0.1307,), (0.3081,)),  # noqa: B008
            ]
        ),
        target_transform: Optional[Any] = None,
        model: Optional[Union[nn.Module, pl.LightningModule]] = None,
    ) -> None:
        """Initialize.

        Args:
            data_dir (Path): Data directory.
                Defaults to package_root_path.
            batch_size (int): batch size. Defaults to 32.
            transform (Optional[Any]): Tranforms for the inputs.
                Defaults to transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            target_transform (Optional[Any]): Transforms for the target. Defaults to None.
            model (Optional[Union[nn.Module, pl.LightningModule]]): Model. Default to None.
        """
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.model = model

    def prepare_data(self: "MNISTDataModule") -> None:
        """Prepare data."""
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self: "MNISTDataModule", stage: Optional[str] = None) -> None:
        """Setup data module."""
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(
                self.data_dir, train=False, transform=self.transform
            )

        if stage == "predict" or stage is None:
            self.mnist_predict = MNIST(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self: "MNISTDataModule") -> DataLoader:
        """Train dataloader."""
        return DataLoader(
            self.mnist_train, batch_size=self.batch_size, num_workers=cpu_count()
        )

    def enc_train_dataloader(
        self: "MNISTDataModule", context: ts.Context
    ) -> DataLoader:
        """Encrypted train dataloader.

        Args:
            context (ts.Context): Tenseal encryption context.

        Returns:
            DataLoader: train dataloader
        """
        enc_mnist_train = EncConvDataset(
            context,
            self.mnist_train,
            self.model.conv1.kernel_size,
            self.model.conv1.stride[0],
        )
        return DataLoader(enc_mnist_train, batch_size=None)

    def val_dataloader(self: "MNISTDataModule") -> DataLoader:
        """Validation dataloader."""
        return DataLoader(
            self.mnist_val, batch_size=self.batch_size, num_workers=cpu_count()
        )

    def enc_val_dataloader(self: "MNISTDataModule", context: ts.Context) -> DataLoader:
        """Encrypted validation dataloader.

        Args:
            context (ts.Context): Tenseal encryption context.

        Returns:
            DataLoader: validation dataloader
        """
        enc_mnist_val = EncConvDataset(
            context,
            self.mnist_val,
            self.model.conv1.kernel_size,
            self.model.conv1.stride[0],
        )
        return DataLoader(enc_mnist_val, batch_size=None)

    def test_dataloader(self: "MNISTDataModule") -> DataLoader:
        """Test dataloader."""
        return DataLoader(
            self.mnist_test, batch_size=self.batch_size, num_workers=cpu_count()
        )

    def enc_test_dataloader(self: "MNISTDataModule", context: ts.Context) -> DataLoader:
        """Encrypted test dataloader.

        Args:
            context (ts.Context): Tenseal encryption context.

        Returns:
            DataLoader: test dataloader
        """
        enc_mnist_test = EncConvDataset(
            context,
            self.mnist_test,
            self.model.conv1.kernel_size,
            self.model.conv1.stride[0],
        )
        return DataLoader(enc_mnist_test, batch_size=None)

    def predict_dataloader(self: "MNISTDataModule") -> DataLoader:
        """Predict dataloader."""
        return DataLoader(
            self.mnist_predict, batch_size=self.batch_size, num_workers=cpu_count()
        )

    def enc_predict_dataloader(
        self: "MNISTDataModule", context: ts.Context
    ) -> DataLoader:
        """Encrypted predict dataloader.

        Args:
            context (ts.Context): Tenseal encryption context.

        Returns:
            DataLoader: predict dataloader
        """
        enc_mnist_predict = EncConvDataset(
            context,
            self.mnist_predict,
            self.model.conv1.kernel_size,
            self.model.conv1.stride[0],
        )
        return DataLoader(enc_mnist_predict, batch_size=None)
