"""Cryptic Sage route."""

from flask import Blueprint
from flask import Response

cryptic_sage_bp = Blueprint("cryptic_sage", __name__, url_prefix="/v1/crypticsage")


@cryptic_sage_bp.post("/")
def cryptic_sage_post() -> Response:
    """Test encryption.

    Returns:
        Response: Flask response
    """
    return "OK"
