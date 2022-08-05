"""Encrypted Convolution Dataset wrapper."""

from typing import Any

import tenseal as ts
from torch.utils.data import Dataset


class EncConvDataset(Dataset):
    """Encrypted Convolution Dataset."""

    def __init__(
        self: "EncConvDataset",
        context: ts.Context,
        dataset: Dataset,
        kernel_shape: tuple,
        stride: int,
        input_shape: tuple = (28, 28),
    ) -> None:
        """Encrypted dataset.

        Args:
            context (ts.Context): Tenseal encryption context.
            dataset (Dataset): Initialized dataset class to wrap.
            kernel_shape (tuple): Shape of the convolution kernel.
            stride (int): Stride length.
            input_shape (tuple): Shape of the input image. Defaults to (28, 28).

        """
        self.context = context
        self.dataset = dataset
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.input_shape = input_shape

    def __len__(self: "EncConvDataset") -> int:
        """Length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return self.dataset.__len__()

    def __getitem__(self: "EncConvDataset", index: int) -> Any:
        """Get item from the dataset.

        Args:
            index (int): index of the item.

        Returns:
            Any: Item
        """
        items = self.dataset.__getitem__(index)
        # Encrypt items
        enc_x, windows_nb = ts.im2col_encoding(
            self.context,
            items[0].view(self.input_shape[0], self.input_shape[1]).tolist(),
            self.kernel_shape[0],
            self.kernel_shape[1],
            self.stride,
        )
        enc_y = ts.ckks_tensor(self.context, [items[1]])
        return tuple([enc_x, enc_y, windows_nb])
