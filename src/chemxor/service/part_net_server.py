"""Partitioned network Service."""

from http import HTTPStatus
from typing import Any, List, Optional, Union

from flask import Blueprint, Flask, make_response, request, Response
from pydantic import parse_raw_as, ValidationError
from pytorch_lightning import LightningModule
import tenseal as ts
from torch import nn

from chemxor.schema.fhe_model import (
    PartFHEModelQueryPostRequest,
    PartFHEModelQueryPostResponse,
)


class PartitionNetServer:
    """Partitioned network service."""

    def __init__(
        self: "PartitionNetServer",
        part_net_list: Optional[List[Union[nn.Module, LightningModule]]] = None,
        flask_app: Optional[Flask] = None,
    ) -> None:
        """Init."""
        self.part_net_list = part_net_list or []
        self.flask_app = flask_app or Flask()
        self.flask_blueprint = Blueprint()

    def forward(self: "PartitionNetServer", x: Any, step: int) -> Any:
        """Forward function.

        Args:
            x (Any): model input
            step (int) : model step

        Returns:
            Any: model output
        """
        x = self.part_net_list[step](x)
        return x

    def service_get(self: "PartitionNetServer") -> Response:
        """Get request handler.

        Returns:
            Response: Flask response
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

        response_body = PartFHEModelQueryPostResponse()
        return make_response(response_body.json(), HTTPStatus.OK)

    def service_post(self: "PartitionNetServer") -> Response:
        """Post request handler.

        Returns:
            Response: Flask response
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

        # Parse context from request
        context = ts.context_from(bytes.fromhex(model_query_request.context))

        # convert enc input to enc vector
        enc_input = ts.ckks_vector_from(
            context, bytes.fromhex(model_query_request.model_input)
        )
        enc_output_tensor = self.forward(enc_input, model_query_request.model_step)

        # determine next step
        next_step = (
            (model_query_request.model_step + 1)
            if (model_query_request.model_step + 1) <= len(self.part_net_list)
            else None
        )
        response_body = PartFHEModelQueryPostResponse(
            output_tensor=enc_output_tensor.serialize().hex(), next_step=next_step
        )
        return make_response(response_body.json(), HTTPStatus.OK)

    def build_service(self: "PartitionNetServer") -> Optional[Flask]:
        """Build Flask app server."""
        return self.flask_app

    def get_flask_app(self: "PartitionNetServer") -> Optional[Flask]:
        """Return Flask app if present."""
        return self.flask_app

    def get_flask_blueprint(self: "PartitionNetServer") -> Optional[Blueprint]:
        """Return Flask blueprint if present."""
        return self.flask_app
