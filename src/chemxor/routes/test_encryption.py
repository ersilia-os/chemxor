"""Test encryption route."""

from http import HTTPStatus

from flask import Blueprint, make_response, request, Response
from pydantic import parse_raw_as, ValidationError
import tenseal as ts

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
        test_enc = parse_raw_as(TestEncPost, request.json)
    except ValidationError:
        response_body = TestEncResponse(vector="")
        return make_response(response_body.json(), HTTPStatus.BAD_REQUEST)
    except Exception:
        response_body = TestEncResponse(vector="")
        return make_response(response_body.json(), HTTPStatus.INTERNAL_SERVER_ERROR)

    # Perform addition
    context = ts.context_from(bytes.fromhex(test_enc.context))
    new_vector = ts.bfv_vector_from(context, bytes.fromhex(test_enc.vector)) + [
        1,
        1,
        1,
        1,
        1,
    ]
    return make_response(
        TestEncResponse(vector=new_vector.serialize().hex()).json(), HTTPStatus.OK
    )
