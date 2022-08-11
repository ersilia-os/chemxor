"""Reusable BLoCs."""

from http import HTTPStatus
from typing import Any, Union

from flask import Blueprint, make_response, request, Response
from pydantic import parse_raw_as, ValidationError
from pytorch_lightning import LightningModule
import tenseal as ts
from torch import nn

from chemxor.schema.fhe_model import (
    ModelContextParams,
    ModelInfo,
    PartFHEModelQueryGetResponse,
    PartFHEModelQueryPostRequest,
    PartFHEModelQueryPostResponse,
)


def process_fhe_model_query(
    query: PartFHEModelQueryPostRequest,
    model: Union[nn.Module, LightningModule],
    step: int,
) -> Any:
    """Evaluate fhe model."""
    # Parse context from request
    context = ts.context_from(bytes.fromhex(query.context))

    # convert enc input to enc vector
    enc_input = ts.ckks_vector_from(context, bytes.fromhex(query.model_input))
    return model(enc_input, step)


def generate_blueprint(
    part_net: Union[nn.Module, LightningModule],
    name: str = "partnet",
    url_prefix: str = "/v1/fhe",
) -> Blueprint:
    """Generate flask blueprint for a specific model.

    Args:
        part_net (Union[nn.Module, LightningModule]): FHE model
        name (str): Name of blueprint
        url_prefix (str): Blueprint url_prefix

    Returns:
        Blueprint: Flask blueprint
    """
    part_net_blueprint = Blueprint(name, name, url_prefix=url_prefix)

    @part_net_blueprint.post("/")
    def part_net_model_post() -> Response:
        """Partnet POST request handler.

        Returns:
            Response: Flask response.
        """
        # Parse json from requests
        try:
            model_query_request = parse_raw_as(
                PartFHEModelQueryPostRequest, request.json
            )
        except ValidationError:
            response_body = PartFHEModelQueryPostResponse()
            return make_response(response_body.json(), HTTPStatus.BAD_REQUEST)
        except Exception:
            response_body = PartFHEModelQueryPostResponse()
            return make_response(response_body.json(), HTTPStatus.INTERNAL_SERVER_ERROR)

        enc_output_tensor = process_fhe_model_query(part_net, model_query_request)

        # determine next step
        next_step = (
            (model_query_request.model_step + 1)
            if (model_query_request.model_step + 1) <= len(part_net)
            else None
        )
        response_body = PartFHEModelQueryPostResponse(
            output_tensor=enc_output_tensor.serialize().hex(),
            next_step=next_step,
            preprocess_next_args=part_net.pre_process[model_query_request.model_step],
        )
        return make_response(response_body.json(), HTTPStatus.OK)

    @part_net_blueprint.get("/")
    def part_net_model_get() -> Response:
        """Partnet GET request handler.

        Returns:
            Response: Flask response.
        """
        response_body = PartFHEModelQueryGetResponse(
            model_info=ModelInfo(
                model_name=str(part_net),
                model_steps=part_net.steps,
                context_params=ModelContextParams(
                    bit_scale=part_net.bit_scale,
                    poly_modulus_degree=part_net.poly_modulus_degree,
                    coeff_mod_bit_sizes=part_net.coeff_mod_bit_sizes,
                ),
            )
        )
        return make_response(response_body.json(), HTTPStatus.OK)

    return part_net_blueprint
