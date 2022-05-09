"""Test encryption route."""

from http.client import OK
import json
from http import HTTPStatus

from flask import Blueprint, make_response, Response, request
from pydantic import parse_obj_as, ValidationError

from chemxor.schema.test_encryption import TestEncPost, TestEncResponse

test_encryption_bp = Blueprint("test_encryption", __name__, url_prefix="/v1/test")


@test_encryption_bp.post("/")
def test_encryption_post() -> Response:
    """Test encryption.

    Returns:
        Response: Flask response
    """
    # Parse json from requests
    try:
        test_enc = parse_obj_as(TestEncPost, json.loads(request.data))
    except ValidationError:
        response_body = TestEncResponse(vector=[])
        return make_response(response_body.json(), HTTPStatus.BAD_REQUEST)
    except Exception:
        response_body = TestEncResponse(vector=[])
        return make_response(response_body.json(), HTTPStatus.INTERNAL_SERVER_ERROR)

    # Perform addition
    new_vector = test_enc.vector + [1, 2, 3]
    return make_response(TestEncResponse(vector=new_vector).json(), HTTPStatus.OK)
