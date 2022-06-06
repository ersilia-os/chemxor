"""FHE for AI/ML models."""

from pathlib import Path
from types import MethodType
from typing import Any, Union

import pytorch_lightning as pl
import tenseal as ts
from torch import nn, Tensor

from chemxor.crypt.onnx_utils import onnx_to_torch_fhe_forward, torch_to_onnx


class FHECryptor:
    """FHE for AI/ML models."""

    def __init__(self: "FHECryptor", context: ts.Context) -> None:
        """Setup cryptographic params.

        Args:
            context (ts.Context): TenSeal Context
        """
        self.context = context

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
        """Convert the model to be FHE compatible.

        Args:
            model (Union[nn.Module, pl.LightningModule]): Model to convert
            dummy_input (Any): Dummy input

        Returns:
            Union[nn.Module, pl.LightningModule]: Converted model
        """
        onnx_model = torch_to_onnx(model, dummy_input)
        modified_forward = onnx_to_torch_fhe_forward(onnx_model, dummy_input)
        model.forward = MethodType(modified_forward, model)
        return model

    def encrypt_tensor(self: "FHECryptor", tensor: Tensor) -> Tensor:
        """Encrypt the tensor.

        Args:
            tensor (Tensor): Tensor to encrypt

        Raises:
            Exception: Public key not found

        Returns:
            Tensor: Encrypted tensor
        """
        if self.context.has_public_key() is False:
            raise Exception("Public key not found in the context")
        else:
            return ts.ckks_tensor(self.context, tensor)

    def decrypt_tensor(self: "FHECryptor", enc_tensor: ts.CKKSTensor) -> Tensor:
        """Decrypt the tensor.

        Args:
            enc_tensor (Tensor): Encrypted tensor

        Raises:
            Exception: Secret key not found

        Returns:
            Tensor: Decrypted tensor
        """
        if self.context.has_secret_key() is False:
            raise Exception("Secret key not found in the context")
        else:
            return enc_tensor.decrypt(self.context.secret_key())
