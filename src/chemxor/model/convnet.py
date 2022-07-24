"""Basic convolution network."""

from typing import Any

import pytorch_lightning as pl
import torch as t
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, Optimizer
from torchmetrics import Accuracy, MetricCollection, Precision, Recall


# Adapted from https://github.dev/OpenMined/TenSEAL/blob/6516f215a0171fd9ad70f60f2f9b3d0c83d0d7c4/tutorials/Tutorial%204%20-%20Encrypted%20Convolution%20on%20MNIST.ipynb
class ConvNet(pl.LightningModule):
    """Cryptic Sage."""

    def __init__(self: "ConvNet", hidden: int = 64, output: int = 10) -> None:
        """Init."""
        super().__init__()
        self.hidden = hidden
        self.output = output
        self.conv1 = nn.Conv2d(1, 28, kernel_size=3, padding=0, stride=3)
        self.fc1 = nn.Linear(2800, hidden)
        self.fc2 = nn.Linear(hidden, output)

        # Metrics
        metrics = MetricCollection(
            [
                Accuracy(),
                Precision(num_classes=output, average="macro"),
                Recall(num_classes=output, average="macro"),
            ]
        )
        self.train_metrics = metrics.clone(prefix="TRAIN_")
        self.valid_metrics = metrics.clone(prefix="VAL_")
        self.test_metrics = metrics.clone(prefix="TEST_")

    def forward(self: "ConvNet", x: Any) -> Any:
        """Forward function.

        Args:
            x (Any): model input

        Returns:
            Any: model output
        """
        x = self.conv1(x)
        # the model uses the square activation function
        x = x * x
        # flattening while keeping the batch axis
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = x * x
        x = self.fc2(x)
        return x

    def training_step(self: "ConvNet", batch: Any, batch_idx: Any) -> Any:
        """Training step.

        Args:
            batch (Any): input batch
            batch_idx (Any): batch index

        Returns:
            Any: training step loss
        """
        x, y = batch
        output = self(x)
        loss = F.cross_entropy(output, y.type(t.long))

        # Logging to TensorBoard by default
        self.log("TRAIN_Loss", loss)
        # Logging metrics
        metrics = self.train_metrics(output, y.type(t.int))
        self.log_dict(metrics)
        return loss

    def validation_step(self: "ConvNet", batch: Any, batch_idx: Any) -> None:
        """Validation step.

        Args:
            batch (Any): input batch
            batch_idx (Any): batch index
        """
        x, y = batch
        output = self(x)
        loss = F.cross_entropy(output, y.type(t.long))
        # Logging to TensorBoard by default
        self.log("VAL_Loss", loss)

        # Logging metrics
        metrics = self.valid_metrics(output, y.type(t.int))
        self.log_dict(metrics)

    def test_step(self: "ConvNet", batch: Any, batch_idx: Any) -> None:
        """Test step.

        Args:
            batch (Any): input batch
            batch_idx (Any): batch index
        """
        x, y = batch
        output = self(x)
        loss = F.cross_entropy(output, y.type(t.long))

        # Logging to TensorBoard by default
        self.log("TEST_Loss", loss)

        # Logging metrics
        metrics = self.test_metrics(output, y.type(t.int))
        self.log_dict(metrics)

    def configure_optimizers(self: "ConvNet") -> Optimizer:
        """Configure optimizer."""
        optimizer = Adam(self.parameters(), lr=0.001)
        return optimizer
