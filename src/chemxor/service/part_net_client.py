"""Partitioned Network Client."""

import json
from typing import Any

from pydantic import parse_obj_as
import requests
import tenseal as ts

from chemxor.schema.fhe_model import (
    ModelInfo,
    PartFHEModelQueryGetResponse,
    PartFHEModelQueryPostRequest,
    PartFHEModelQueryPostResponse,
)
from chemxor.utils import prepare_fhe_input, smiles_to_imcol


class PartitionNetClient:
    """Partitioned network client."""

    def __init__(self: "PartitionNetClient", url: str) -> None:
        """Initialize Client.

        Args:
            url (str): URL of the model service
        """
        self.model_url = url
        self.model_info = self.retrieve_model_info(url)
        self.enc_context = self.create_ts_context_from_model()
        self.public_context = self.enc_context.copy()
        self.public_context.make_context_public()

    def create_ts_context_from_model(
        self: "PartitionNetClient", model_info: ModelInfo
    ) -> ts.Context:
        """Create tenseal context from model info."""
        bits_scale = model_info.context_params.bit_scale
        enc_context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=model_info.context_params.poly_modulus_degree,
            coeff_mod_bit_sizes=model_info.context_params.coeff_mod_bit_sizes,
        )
        enc_context.global_scale = pow(2, bits_scale)
        enc_context.generate_galois_keys()
        return enc_context

    def retrieve_model_info(self: "PartitionNetClient", url: str) -> Any:
        """Retrieve model information to create encryption context and prepare input.

        Args:
            url (str): URL of the model service

        Returns:
            Any: Model Info
        """
        response = requests.get(self.model_url)
        response = parse_obj_as(
            PartFHEModelQueryGetResponse, json.loads(response.content)
        )
        return response.model_info

    def query(self: "PartitionNetClient", x: str) -> Any:
        """Query a FHE model service.

        Args:
            x (str): SMILES string

        Returns:
            Any: Model output
        """
        output = smiles_to_imcol(x, self.enc_context)
        for step in range(self.model_info.steps + 1):
            request = PartFHEModelQueryPostRequest(
                ts_context=self.public_context.serialize().hex(),
                model_input=output,
                model_step=step,
            )
            response = requests.post(self.model_url, json=request.json())
            response = parse_obj_as(
                PartFHEModelQueryPostResponse, json.loads(response.content)
            )
            output = ts.ckks_vector_from(
                self.enc_context, bytes.fromhex(response.model_output)
            )
            dec_out = output.decrypt()
            output = prepare_fhe_input(
                dec_out, response.preprocess_next_args, self.enc_context
            )
        return output.decrypt()
