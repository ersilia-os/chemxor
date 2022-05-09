"""Status route."""

from flask import Blueprint
from flask import Response

status_bp = Blueprint("status", __name__, url_prefix="/v1/status")


@status_bp.get("/")
def status_ok() -> Response:
    """Status route.

    Returns:
        Response: Flask response
    """
    return "OK"
