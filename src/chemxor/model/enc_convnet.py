"""Encrypted ConvNet."""

from typing import Any

import pytorch_lightning as pl
import tenseal as ts
from torch.nn import functional as F
from torch.optim import Adam, Optimizer
from torchmetrics import Accuracy, ConfusionMatrix, MetricCollection, Precision, Recall


from chemxor.model.convnet import ConvNet


# Adapted from https://github.dev/OpenMined/TenSEAL/blob/6516f215a0171fd9ad70f60f2f9b3d0c83d0d7c4/tutorials/Tutorial%204%20-%20Encrypted%20Convolution%20on%20MNIST.ipynb
class EncryptedConvNet(pl.LightningModule):
    """Encrypted ConvNet."""

    def __init__(self: "EncryptedConvNet", model: ConvNet) -> None:
        """Init."""
        super().__init__()

        self.conv1_weight = model.conv1.weight.data.view(
            model.conv1.out_channels,
            model.conv1.kernel_size[0],
            model.conv1.kernel_size[1],
        ).tolist()
        self.conv1_bias = model.conv1.bias.data.tolist()

        self.fc1_weight = model.fc1.weight.T.data.tolist()
        self.fc1_bias = model.fc1.bias.data.tolist()

        self.fc2_weight = model.fc2.weight.T.data.tolist()
        self.fc2_bias = model.fc2.bias.data.tolist()

        # Metrics
        metrics = MetricCollection(
            [
                Accuracy(),
                Precision(num_classes=model.output, average="macro"),
                Recall(num_classes=model.output, average="macro"),
                ConfusionMatrix(num_classes=model.output),
            ]
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.valid_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def forward(self: "EncryptedConvNet", x: Any, windows_nb: int) -> Any:
        """Forward function.

        Args:
            x (Any): model input
            windows_nb (int): window size.

        Returns:
            Any: model output
        """
        # conv layer
        enc_channels = []
        for kernel, bias in zip(self.conv1_weight, self.conv1_bias):
            y = x.conv2d_im2col(kernel, windows_nb) + bias
            enc_channels.append(y)
        # pack all channels into a single flattened vector
        enc_x = ts.CKKSVector.pack_vectors(enc_channels)
        # square activation
        enc_x.square_()
        # fc1 layer
        enc_x = enc_x.mm(self.fc1_weight) + self.fc1_bias
        # square activation
        enc_x.square_()
        # fc2 layer
        enc_x = enc_x.mm(self.fc2_weight) + self.fc2_bias
        return enc_x

    def training_step(self: "ConvNet", batch: Any, batch_idx: Any) -> Any:
        """Training step.

        Args:
            batch (Any): input batch
            batch_idx (Any): batch index

        Returns:
            Any: training step loss
        """
        x, y, windows_nb = batch
        output = self(x, windows_nb)
        loss = F.cross_entropy(output, y)

        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        # Logging metrics
        metrics = self.train_metrics(output, y)
        self.log_dict(metrics)
        return loss

    def validation_step(self: "ConvNet", batch: Any, batch_idx: Any) -> None:
        """Validation step.

        Args:
            batch (Any): input batch
            batch_idx (Any): batch index
        """
        x, y, windows_nb = batch
        output = self(x, windows_nb)
        loss = F.cross_entropy(output, y)

        # Logging to TensorBoard by default
        self.log("validation_loss", loss)

        # Logging metrics
        metrics = self.valid_metrics(output, y)
        self.log_dict(metrics)

    def test_step(self: "ConvNet", batch: Any, batch_idx: Any) -> None:
        """Test step.

        Args:
            batch (Any): input batch
            batch_idx (Any): batch index
        """
        x, y, windows_nb = batch
        output = self(x, windows_nb)
        loss = F.cross_entropy(output, y)

        # Logging to TensorBoard by default
        self.log("test_loss", loss)

        # Logging metrics
        metrics = self.test_metrics(output, y)
        self.log_dict(metrics)

    def configure_optimizers(self: "ConvNet") -> Optimizer:
        """Configure optimizer."""
        optimizer = Adam(self.parameters(), lr=0.001)
        return optimizer
