"""Basic convolution network."""

from typing import Any, Callable

import pytorch_lightning as pl
import torch as t
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, Optimizer
from torchmetrics import Accuracy, MetricCollection, Precision, Recall


class ConvLinearZero(pl.LightningModule):
    """Fast distilled model."""

    def __init__(
        self: "ConvLinearZero", output: int = 10, criterion: Callable = F.cross_entropy
    ) -> None:
        """Init."""
        super().__init__()
        self.output = output
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=0, stride=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=0, stride=3)
        self.fc1 = nn.Linear(3200, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, output)

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

        # Criterion
        self.criterion = criterion

    def forward(self: "ConvLinearZero", x: Any) -> Any:
        """Forward function.

        Args:
            x (Any): model input

        Returns:
            Any: model output
        """
        x = self.conv1(x)
        x = x * x
        x = self.conv2(x)
        x = x * x

        # flattening while keeping the batch axis
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = x * x
        x = self.fc2(x)
        x = x * x
        x = self.fc3(x)
        x = x * x
        x = self.fc4(x)
        return x

    def training_step(self: "ConvLinearZero", batch: Any, batch_idx: Any) -> Any:
        """Training step.

        Args:
            batch (Any): input batch
            batch_idx (Any): batch index

        Returns:
            Any: training step loss
        """
        x, y = batch
        output = self(x)
        loss = self.criterion(output, y.type(t.long))

        # Logging to TensorBoard by default
        self.log("TRAIN_Loss", loss)
        # Logging metrics
        metrics = self.train_metrics(output, y.type(t.int))
        self.log_dict(metrics)
        return loss

    def validation_step(self: "ConvLinearZero", batch: Any, batch_idx: Any) -> None:
        """Validation step.

        Args:
            batch (Any): input batch
            batch_idx (Any): batch index
        """
        x, y = batch
        output = self(x)
        loss = self.criterion(output, y.type(t.long))
        # Logging to TensorBoard by default
        self.log("VAL_Loss", loss)

        # Logging metrics
        metrics = self.valid_metrics(output, y.type(t.int))
        self.log_dict(metrics)

    def test_step(self: "ConvLinearZero", batch: Any, batch_idx: Any) -> None:
        """Test step.

        Args:
            batch (Any): input batch
            batch_idx (Any): batch index
        """
        x, y = batch
        output = self(x)
        loss = self.criterion(output, y.type(t.long))

        # Logging to TensorBoard by default
        self.log("TEST_Loss", loss)

        # Logging metrics
        metrics = self.test_metrics(output, y.type(t.int))
        self.log_dict(metrics)

    def configure_optimizers(self: "ConvLinearZero") -> Optimizer:
        """Configure optimizer."""
        optimizer = Adam(self.parameters(), lr=0.001)
        return optimizer


class ConvLinearOne(pl.LightningModule):
    """Distilled model one."""

    def __init__(
        self: "ConvLinearZero", output: int = 10, criterion: Callable = F.cross_entropy
    ) -> None:
        """Init."""
        super().__init__()
        self.output = output
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=0, stride=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=0, stride=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=0, stride=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=0, stride=1)
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, output)

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

        # Criterion
        self.criterion = criterion

    def forward(self: "ConvLinearZero", x: Any) -> Any:
        """Forward function.

        Args:
            x (Any): model input

        Returns:
            Any: model output
        """
        x = self.conv1(x)
        x = x * x
        x = self.conv2(x)
        x = x * x
        x = self.conv3(x)
        x = x * x
        x = self.conv4(x)
        x = x * x

        # flattening while keeping the batch axis
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = x * x
        x = self.fc2(x)
        x = x * x
        x = self.fc3(x)
        x = x * x
        x = self.fc4(x)
        return x

    def training_step(self: "ConvLinearZero", batch: Any, batch_idx: Any) -> Any:
        """Training step.

        Args:
            batch (Any): input batch
            batch_idx (Any): batch index

        Returns:
            Any: training step loss
        """
        x, y = batch
        output = self(x)
        loss = self.criterion(output, y.type(t.long))

        # Logging to TensorBoard by default
        self.log("TRAIN_Loss", loss)
        # Logging metrics
        metrics = self.train_metrics(output, y.type(t.int))
        self.log_dict(metrics)
        return loss

    def validation_step(self: "ConvLinearZero", batch: Any, batch_idx: Any) -> None:
        """Validation step.

        Args:
            batch (Any): input batch
            batch_idx (Any): batch index
        """
        x, y = batch
        output = self(x)
        loss = self.criterion(output, y.type(t.long))
        # Logging to TensorBoard by default
        self.log("VAL_Loss", loss)

        # Logging metrics
        metrics = self.valid_metrics(output, y.type(t.int))
        self.log_dict(metrics)

    def test_step(self: "ConvLinearZero", batch: Any, batch_idx: Any) -> None:
        """Test step.

        Args:
            batch (Any): input batch
            batch_idx (Any): batch index
        """
        x, y = batch
        output = self(x)
        loss = self.criterion(output, y.type(t.long))

        # Logging to TensorBoard by default
        self.log("TEST_Loss", loss)

        # Logging metrics
        metrics = self.test_metrics(output, y.type(t.int))
        self.log_dict(metrics)

    def configure_optimizers(self: "ConvLinearZero") -> Optimizer:
        """Configure optimizer."""
        optimizer = Adam(self.parameters(), lr=0.001)
        return optimizer
