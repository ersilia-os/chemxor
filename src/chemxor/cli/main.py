"""Chemxor CLI."""

import click
from flask import Flask
import requests
import tenseal as ts

from chemxor.routes.cryptic_sage import cryptic_sage_bp
from chemxor.routes.status import status_bp
from chemxor.routes.test_encryption import test_encryption_bp
from chemxor.schema.test_encryption import TestEncPost, TestEncResponse
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
    context = ts.context(
        ts.SCHEME_TYPE.BFV, poly_modulus_degree=4096, plain_modulus=1032193
    )
    sk = context.secret_key()
    context.make_context_public()
    plain_vector = [60, 66, 73, 81, 90]
    encrypted_vector = ts.bfv_vector(context, plain_vector)
    test_enc_post = TestEncPost(context=context, vector=encrypted_vector)
    res = requests.post("localhost:7880/v1/test_encryption/", json=test_enc_post.json())
    print(res.json())
    test_enc_res = TestEncResponse.parse_obj(res.json())
    dec_vector = test_enc_res.vector.decrypt()
    print(f"Decrypted vector: {dec_vector}")


@click.group()
@click.version_option(version=__version__)
def main() -> None:
    """Chemxor console."""
    pass


main.add_command(serve)
main.add_command(query)
