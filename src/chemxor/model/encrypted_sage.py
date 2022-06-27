"""Encrypted Sage."""

from typing import Any

import pytorch_lightning as pl
import tenseal as ts
import torch
from torch.nn import functional as F
from torch.optim import Adam, Optimizer

from chemxor.model.cryptic_sage import CrypticSage


class EncryptedSage(pl.LightningModule):
    """Encrypted Sage."""

    def __init__(self: "EncryptedSage", model: CrypticSage) -> None:
        """Init."""
        super().__init__()

        self.layer_1_weight = model.layer_1.weight.T.data.tolist()
        self.layer_1_bias = model.layer_1.bias.data.tolist()

        self.layer_2_weight = model.layer_2.weight.T.data.tolist()
        self.layer_2_bias = model.layer_2.bias.data.tolist()

        self.layer_3_weight = model.layer_3.weight.T.data.tolist()
        self.layer_3_bias = model.layer_3.bias.data.tolist()

        self.layer_4_weight = model.layer_4.weight.T.data.tolist()
        self.layer_4_bias = model.layer_4.bias.data.tolist()

        self.layer_5_weight = model.layer_5.weight.T.data.tolist()
        self.layer_5_bias = model.layer_5.bias.data.tolist()

    def forward(self: "EncryptedSage", x: Any) -> Any:
        """Forward function.

        Args:
            x (Any): model input

        Returns:
            Any: model output
        """
        # Check for tensor type as encrypted tensors do not impement x.size and x.view
        if type(x) not in [
            ts.CKKSTensor,
            ts.BFVTensor,
            ts.CKKSVector,
            ts.BFVVector,
            ts.PlainTensor,
        ]:
            batch_size = x.size(0)
            x = x.view(batch_size, -1)
            converter = torch.tensor

        else:

            def converter(x: Any) -> Any:
                return x

        # layer 1
        x = x.mm(converter(self.layer_1_weight)) + converter(self.layer_1_bias)
        x.square_()

        # layer 2
        x = x.mm(converter(self.layer_2_weight)) + converter(self.layer_2_bias)
        x.square_()

        # layer 3
        x = x.mm(converter(self.layer_3_weight)) + converter(self.layer_3_bias)
        x.square_()

        # layer 4
        x = x.mm(converter(self.layer_4_weight)) + converter(self.layer_4_bias)
        x.square_()

        # layer 5
        x = x.mm(converter(self.layer_5_weight)) + converter(self.layer_5_bias)

        return x

    def training_step(self: "EncryptedSage", batch: Any, batch_idx: Any) -> Any:
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

    def validation_step(self: "EncryptedSage", batch: Any, batch_idx: Any) -> None:
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

    def test_step(self: "EncryptedSage", batch: Any, batch_idx: Any) -> None:
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

    def configure_optimizers(self: "EncryptedSage") -> Optimizer:
        """Configure optimizer."""
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer
