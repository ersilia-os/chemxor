"""Test encryption route."""

from flask import Blueprint
from flask import Response

test_encryption_bp = Blueprint("test_encryption", __name__, url_prefix="/v1/test")


@test_encryption_bp.post("/")
def test_encryption_post() -> Response:
    """Test encryption.

    Returns:
        Response: Flask response
    """
    return "OK"
