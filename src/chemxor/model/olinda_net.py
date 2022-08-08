"""OlindaNet modules."""

from typing import Any, Callable, Optional

import pytorch_lightning as pl
import tenseal as ts
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, Optimizer
from torchmetrics import Accuracy, MeanSquaredError, MetricCollection, Precision, Recall

from chemxor.model.fhe_activation import softplus_polyval


class OlindaNetZero(pl.LightningModule):
    """OlindaNet Zero: Slim(relatively) distillation network."""

    def __init__(
        self: "OlindaNetZero", output: int = 10, criterion: Optional[Callable] = None
    ) -> None:
        """Init."""
        super().__init__()
        self.output = output
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=0, stride=3)
        self.fc1 = nn.Linear(3200, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output)

        # Classification Metrics
        if output > 1:
            metrics = MetricCollection(
                [
                    Accuracy(),
                    Precision(num_classes=output, average="macro"),
                    Recall(num_classes=output, average="macro"),
                ]
            )
        else:
            metrics = MetricCollection([MeanSquaredError()])

        self.train_metrics = metrics.clone(prefix="TRAIN_")
        self.valid_metrics = metrics.clone(prefix="VAL_")
        self.test_metrics = metrics.clone(prefix="TEST_")

        # Criterion
        if criterion is None:
            if output > 1:
                self.criterion = F.cross_entropy
            else:
                self.criterion = F.mse_loss
        else:
            self.criterion = criterion

        # Encryption context
        bits_scale = 26
        self.enc_context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[
                31,
                bits_scale,
                bits_scale,
                bits_scale,
                bits_scale,
                bits_scale,
                bits_scale,
                31,
            ],
        )
        self.enc_context.global_scale = pow(2, bits_scale)
        self.enc_context.generate_galois_keys()

    def forward(self: "OlindaNetZero", x: Any) -> Any:
        """Forward function.

        Args:
            x (Any): model input

        Returns:
            Any: model output
        """
        x = self.conv1(x)
        x = softplus_polyval(x)

        # flattening while keeping the batch axis
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = softplus_polyval(x)
        x = self.fc2(x)
        x = softplus_polyval(x)
        x = self.fc3(x)
        return x

    def training_step(self: "OlindaNetZero", batch: Any, batch_idx: Any) -> Any:
        """Training step.

        Args:
            batch (Any): input batch
            batch_idx (Any): batch index

        Returns:
            Any: training step loss
        """
        x, y = batch
        output = self(x)
        loss = self.criterion(output, y)

        # Logging to TensorBoard by default
        self.log("TRAIN_Loss", loss)

        # Logging metrics
        if self.output > 1:
            y = y.int()
        metrics = self.train_metrics(output, y)
        self.log_dict(metrics)
        return loss

    def validation_step(self: "OlindaNetZero", batch: Any, batch_idx: Any) -> None:
        """Validation step.

        Args:
            batch (Any): input batch
            batch_idx (Any): batch index
        """
        x, y = batch
        output = self(x)
        loss = self.criterion(output, y)

        # Logging to TensorBoard by default
        self.log("VAL_Loss", loss)

        # Logging metrics
        if self.output > 1:
            y = y.int()
        metrics = self.valid_metrics(output, y)
        self.log_dict(metrics)

    def test_step(self: "OlindaNetZero", batch: Any, batch_idx: Any) -> None:
        """Test step.

        Args:
            batch (Any): input batch
            batch_idx (Any): batch index
        """
        x, y = batch
        output = self(x)
        loss = self.criterion(output, y)

        # Logging to TensorBoard by default
        self.log("TEST_Loss", loss)

        # Logging metrics
        if self.output > 1:
            y = y.int()
        metrics = self.test_metrics(output, y)
        self.log_dict(metrics)

    def configure_optimizers(self: "OlindaNetZero") -> Optimizer:
        """Configure optimizer."""
        optimizer = Adam(self.parameters(), lr=0.001)
        return optimizer


class OlindaNet(pl.LightningModule):
    """OlindaNet: A good compromise(relatively) distillation network."""

    def __init__(
        self: "OlindaNet", output: int = 10, criterion: Optional[Callable] = None
    ) -> None:
        """Init."""
        super().__init__()
        self.output = output
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=0, stride=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=0, stride=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, output)

        # Classification Metrics
        if output > 1:
            metrics = MetricCollection(
                [
                    Accuracy(),
                    Precision(num_classes=output, average="macro"),
                    Recall(num_classes=output, average="macro"),
                ]
            )
        else:
            metrics = MetricCollection([MeanSquaredError()])

        self.train_metrics = metrics.clone(prefix="TRAIN_")
        self.valid_metrics = metrics.clone(prefix="VAL_")
        self.test_metrics = metrics.clone(prefix="TEST_")

        # Criterion
        if criterion is None:
            if output > 1:
                self.criterion = F.cross_entropy
            else:
                self.criterion = F.mse_loss
        else:
            self.criterion = criterion

        # Encryption context
        bits_scale = 26
        self.enc_context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[
                31,
                bits_scale,
                bits_scale,
                bits_scale,
                bits_scale,
                bits_scale,
                bits_scale,
                31,
            ],
        )
        self.enc_context.global_scale = pow(2, bits_scale)
        self.enc_context.generate_galois_keys()

    def forward(self: "OlindaNet", x: Any) -> Any:
        """Forward function.

        Args:
            x (Any): model input

        Returns:
            Any: model output
        """
        x = self.conv1(x)
        x = softplus_polyval(x)
        x = self.conv2(x)
        x = softplus_polyval(x)
        x = self.conv3(x)
        x = softplus_polyval(x)
        x = self.conv4(x)
        x = softplus_polyval(x)

        # flattening while keeping the batch axis
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = softplus_polyval(x)
        x = self.fc2(x)
        x = softplus_polyval(x)
        x = self.fc3(x)
        x = softplus_polyval(x)
        x = self.fc4(x)
        return x

    def training_step(self: "OlindaNet", batch: Any, batch_idx: Any) -> Any:
        """Training step.

        Args:
            batch (Any): input batch
            batch_idx (Any): batch index

        Returns:
            Any: training step loss
        """
        x, y = batch
        output = self(x)
        loss = self.criterion(output, y)

        # Logging to TensorBoard by default
        self.log("TRAIN_Loss", loss)

        # Logging metrics
        if self.output > 1:
            y = y.int()
        metrics = self.train_metrics(output, y)
        self.log_dict(metrics)
        return loss

    def validation_step(self: "OlindaNet", batch: Any, batch_idx: Any) -> None:
        """Validation step.

        Args:
            batch (Any): input batch
            batch_idx (Any): batch index
        """
        x, y = batch
        output = self(x)
        loss = self.criterion(output, y)

        # Logging to TensorBoard by default
        self.log("VAL_Loss", loss)

        # Logging metrics
        if self.output > 1:
            y = y.int()
        metrics = self.valid_metrics(output, y)
        self.log_dict(metrics)

    def test_step(self: "OlindaNet", batch: Any, batch_idx: Any) -> None:
        """Test step.

        Args:
            batch (Any): input batch
            batch_idx (Any): batch index
        """
        x, y = batch
        output = self(x)
        loss = self.criterion(output, y)

        # Logging to TensorBoard by default
        self.log("TEST_Loss", loss)

        # Logging metrics
        if self.output > 1:
            y = y.int()
        metrics = self.test_metrics(output, y)
        self.log_dict(metrics)

    def configure_optimizers(self: "OlindaNet") -> Optimizer:
        """Configure optimizer."""
        optimizer = Adam(self.parameters(), lr=0.001)
        return optimizer


class OlindaNetOne(pl.LightningModule):
    """OlindaNet One: Heavy(relatively) distillation network."""

    def __init__(
        self: "OlindaNetOne", output: int = 10, criterion: Optional[Callable] = None
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

        # Classification Metrics
        if output > 1:
            metrics = MetricCollection(
                [
                    Accuracy(),
                    Precision(num_classes=output, average="macro"),
                    Recall(num_classes=output, average="macro"),
                ]
            )
        else:
            metrics = MetricCollection([MeanSquaredError()])

        self.train_metrics = metrics.clone(prefix="TRAIN_")
        self.valid_metrics = metrics.clone(prefix="VAL_")
        self.test_metrics = metrics.clone(prefix="TEST_")

        # Criterion
        if criterion is None:
            if output > 1:
                self.criterion = F.cross_entropy
            else:
                self.criterion = F.mse_loss
        else:
            self.criterion = criterion

        # Encryption context
        bits_scale = 26
        self.enc_context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[
                31,
                bits_scale,
                bits_scale,
                bits_scale,
                bits_scale,
                bits_scale,
                bits_scale,
                31,
            ],
        )
        self.enc_context.global_scale = pow(2, bits_scale)
        self.enc_context.generate_galois_keys()

    def forward(self: "OlindaNetOne", x: Any) -> Any:
        """Forward function.

        Args:
            x (Any): model input

        Returns:
            Any: model output
        """
        x = self.conv1(x)
        x = softplus_polyval(x)
        x = self.conv2(x)
        x = softplus_polyval(x)
        x = self.conv3(x)
        x = softplus_polyval(x)
        x = self.conv4(x)
        x = softplus_polyval(x)

        # flattening while keeping the batch axis
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = softplus_polyval(x)
        x = self.fc2(x)
        x = softplus_polyval(x)
        x = self.fc3(x)
        x = softplus_polyval(x)
        x = self.fc4(x)
        return x

    def training_step(self: "OlindaNetOne", batch: Any, batch_idx: Any) -> Any:
        """Training step.

        Args:
            batch (Any): input batch
            batch_idx (Any): batch index

        Returns:
            Any: training step loss
        """
        x, y = batch
        output = self(x)
        loss = self.criterion(output, y)

        # Logging to TensorBoard by default
        self.log("TRAIN_Loss", loss)

        # Logging metrics
        if self.output > 1:
            y = y.int()
        metrics = self.train_metrics(output, y)
        self.log_dict(metrics)
        return loss

    def validation_step(self: "OlindaNetOne", batch: Any, batch_idx: Any) -> None:
        """Validation step.

        Args:
            batch (Any): input batch
            batch_idx (Any): batch index
        """
        x, y = batch
        output = self(x)
        loss = self.criterion(output, y)

        # Logging to TensorBoard by default
        self.log("VAL_Loss", loss)

        # Logging metrics
        if self.output > 1:
            y = y.int()
        metrics = self.valid_metrics(output, y)
        self.log_dict(metrics)

    def test_step(self: "OlindaNetOne", batch: Any, batch_idx: Any) -> None:
        """Test step.

        Args:
            batch (Any): input batch
            batch_idx (Any): batch index
        """
        x, y = batch
        output = self(x)
        loss = self.criterion(output, y)

        # Logging to TensorBoard by default
        self.log("TEST_Loss", loss)

        # Logging metrics
        if self.output > 1:
            y = y.int()
        metrics = self.test_metrics(output, y)
        self.log_dict(metrics)

    def configure_optimizers(self: "OlindaNetOne") -> Optimizer:
        """Configure optimizer."""
        optimizer = Adam(self.parameters(), lr=0.001)
        return optimizer
