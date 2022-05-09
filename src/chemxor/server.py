"""Server Factory."""

from typing import Callable

from flask import Flask
from logzero import logger


def create_app(register_bp: Callable[[Flask], None]) -> Flask:
    """Create a Flask App.

    Args:
        register_bp (Callable[[Flask], None]): Function to register Blueprints

    Returns:
        Flask: Flask app
    """
    logger.info("Creating a flask app...")
    app = Flask("Chemxor")
    logger.info(f"Flask app created: {app.import_name}")

    # Register blueprints
    register_bp(app)

    return app
