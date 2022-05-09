"""Chemxor CLI."""

import click
from flask import Flask
import requests

from chemxor.routes.cryptic_sage import cryptic_sage_bp
from chemxor.routes.status import status_bp
from chemxor.routes.test_encryption import test_encryption_bp
from chemxor.server import create_app
from .. import __version__


def register_bp(app: Flask) -> None:
    """Register blueprints."""
    app.register_blueprint(status_bp)
    app.register_blueprint(test_encryption_bp)
    app.register_blueprint(cryptic_sage_bp)


@click.command()
def serve() -> None:
    """Serve models."""
    app = create_app(register_bp)
    app.run(host="localhost", port="7880")


@click.command()
def query() -> None:
    """Query models."""
    pass


@click.group()
@click.version_option(version=__version__)
def main() -> None:
    """Chemxor console."""
    pass


main.add_command(serve)
main.add_command(query)
