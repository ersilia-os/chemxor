"""Service functions."""

from http import HTTPStatus
from typing import Optional, Union

from flask import Blueprint, Flask, make_response, request, Response
from pydantic import parse_raw_as, ValidationError
import pytorch_lightning as pl
import tenseal as ts
from torch import nn

from chemxor.crypt import FHECryptor
from chemxor.schema.fhe_model import FHEModelQueryPostRequest, FHEModelQueryPostResponse
from chemxor.server import create_app


def create_model_server(
    model: Union[nn.Module, pl.LightningModule], prefix: Optional[str]
) -> Flask:
    """Create a flask app to serve FHE models.

    Args:
        model (Union[nn.Module, pl.LightningModule]): Model
        prefix (str): URL prefix for model URL

    Returns:
        Flask: Flask app
    """
    if prefix is None:
        prefix = f"/v1/{model._get_name().lower()}"
    model_bp = Blueprint("cryptic_sage", __name__, url_prefix=prefix)

    @model_bp.post("/")
    def model_post() -> Response:
        """Model post route.

        Returns:
            Response: Flask response
        """
        # Parse json from requests
        try:
            model_query_request = parse_raw_as(FHEModelQueryPostRequest, request.json)
        except ValidationError:
            response_body = FHEModelQueryPostResponse(output_tensor=None)
            return make_response(response_body.json(), HTTPStatus.BAD_REQUEST)
        except Exception:
            response_body = FHEModelQueryPostResponse(output_tensor=None)
            return make_response(response_body.json(), HTTPStatus.INTERNAL_SERVER_ERROR)

        # Parse context from request
        context = ts.context_from(bytes.fromhex(model_query_request.context))

        # initialize fhe cryptor with context
        fhe_cryptor = FHECryptor(context)

        # convert model
        converted_model = fhe_cryptor.convert_model(model)

        # set model to eval
        converted_model.eval()

        # convert enc input to enc tensor
        enc_input_tensor = ts.tensor_from(
            bytes.fromhex(model_query_request.input_tensor)
        )
        enc_output_tensor = converted_model(enc_input_tensor)

        response_body = FHEModelQueryPostResponse(
            output_tensor=enc_output_tensor.serialize().hex()
        )
        return make_response(response_body.json(), HTTPStatus.OK)

    app = create_app()
    app.register_blueprint(model_bp)
    return app
