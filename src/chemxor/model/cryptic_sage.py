"""Cryptic Sage."""

from typing import Any

import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, Optimizer


class CrypticSage(pl.LightningModule):
    """Cryptic Sage."""

    def __init__(self: "CrypticSage") -> None:
        """Init."""
        super().__init__()

        self.layer_1 = nn.Linear(5000, 3000)
        self.layer_2 = nn.Linear(3000, 1000)
        self.layer_3 = nn.Linear(1000, 500)
        self.layer_4 = nn.Linear(500, 100)
        self.layer_5 = nn.Linear(100, 2)

    def forward(self: "CrypticSage", x: Any) -> Any:
        """Forward function.

        Args:
            x (Any): model input

        Returns:
            Any: model output
        """
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        x = F.relu(self.layer_4(x))
        return self.layer_5(x)

    def training_step(self: "CrypticSage", batch: Any, batch_idx: Any) -> Any:
        """Training step.

        Args:
            batch (Any): input batch
            batch_idx (Any): batch index

        Returns:
            Any: training step loss
        """
        x, y = batch
        output = self(x)

        # Logging accuracy
        correct = output.argmax(dim=1).eq(y.argmax(dim=1)).sum().item()
        total = len(y)
        self.log("train_accuracy", correct / total)

        loss = F.cross_entropy(output, y)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self: "CrypticSage", batch: Any, batch_idx: Any) -> None:
        """Validation step.

        Args:
            batch (Any): input batch
            batch_idx (Any): batch index
        """
        x, y = batch
        output = self(x)
        loss = F.cross_entropy(output, y)

        # Logging accuracy
        correct = output.argmax(dim=1).eq(y.argmax(dim=1)).sum().item()
        total = len(y)
        self.log("val_accuracy", correct / total)

        # Logging to TensorBoard by default
        self.log("validation_loss", loss)

    def test_step(self: "CrypticSage", batch: Any, batch_idx: Any) -> None:
        """Test step.

        Args:
            batch (Any): input batch
            batch_idx (Any): batch index
        """
        x, y = batch
        output = self(x)
        loss = F.cross_entropy(output, y)

        # Logging accuracy
        correct = output.argmax(dim=1).eq(y.argmax(dim=1)).sum().item()
        total = len(y)
        self.log("test_accuracy", correct / total)

        # Logging to TensorBoard by default
        self.log("test_loss", loss)

    def configure_optimizers(self: "CrypticSage") -> Optimizer:
        """Configure optimizer."""
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer
