"""Encrypted Dataset wrapper."""

from typing import Any

import tenseal as ts
from torch.utils.data import Dataset


class EncDataset(Dataset):
    """Encrypted Dataset."""

    def __init__(self: "EncDataset", context: ts.Context, dataset: Dataset) -> None:
        """Encrypted dataset.

        Args:
            context (ts.Context): Tenseal encryption context.
            dataset (Dataset): Initialized dataset class to wrap.

        """
        self.context = context
        self.dataset = dataset

    def __len__(self: "EncDataset") -> int:
        """Length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return self.dataset.__len__()

    def __getitem__(self: "EncDataset", index: int) -> Any:
        """Get item from the dataset.

        Args:
            index (int): index of the item.

        Returns:
            Any: Item
        """
        items = self.dataset.__getitem__(index)
        # Encrypt items
        enc_item_list = []
        for item in items:
            try:
                _ = self.context.global_scale
                enc_item_list.append(ts.ckks_tensor(self.context, item.reshape(1, -1)))
            except ValueError:
                enc_item_list.append(ts.bfv_tensor(self.context, item.reshape(1, -1)))
        return tuple(enc_item_list)
