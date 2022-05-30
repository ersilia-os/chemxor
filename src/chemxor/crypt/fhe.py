"""FHE for AI/ML models."""

from pathlib import Path
from types import MethodType
from typing import Any, Union

import pytorch_lightning as pl
from torch import nn, Tensor

from chemxor.crypt.onnx_utils import onnx_to_torch_fhe_forward, torch_to_onnx


class FHECryptor:
    """FHE for AI/ML models."""

    def __init__(self: "FHECryptor", context: Any) -> None:
        """Setup cryptographic params."""
        pass

    def load_model(
        self: "FHECryptor", model: bytes
    ) -> Union[nn.Module, pl.LightningModule]:
        """Load the model."""
        pass

    def save_model(
        self: "FHECryptor", path: Path
    ) -> Union[nn.Module, pl.LightningModule]:
        """Save the model."""
        pass

    def convert_model(
        self: "FHECryptor",
        model: Union[nn.Module, pl.LightningModule],
        dummy_input: Any,
    ) -> Union[nn.Module, pl.LightningModule]:
        """Encrypt the model."""
        onnx_model = torch_to_onnx(model, dummy_input)
        modified_forward = onnx_to_torch_fhe_forward(onnx_model, dummy_input)
        model.forward = MethodType(modified_forward, model)
        return model

    def encrypt_tensor(self: "FHECryptor", tensor: Tensor) -> Tensor:
        """Encrypt the tensor."""
        pass

    def decrypt_tensor(self: "FHECryptor", tensor: Tensor) -> Tensor:
        """Decrypt the tensor."""
        pass
