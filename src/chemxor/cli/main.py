"""Chemxor CLI."""

import click
import json
from flask import Flask
from pydantic import parse_obj_as
import requests
import tenseal as ts

from chemxor.cli.partnet import partnet
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
    local_context = context.copy()
    context.make_context_public()
    plain_vector = [60, 66, 73, 81, 90]
    encrypted_vector = ts.bfv_vector(context, plain_vector)
    test_enc_post = TestEncPost(
        context=context.serialize().hex(), vector=encrypted_vector.serialize().hex()
    )
    res = requests.post("http://localhost:7880/v1/test/", json=test_enc_post.json())
    test_enc_res = parse_obj_as(TestEncResponse, json.loads(res.content))
    dec_vector = ts.bfv_vector_from(local_context, bytes.fromhex(test_enc_res.vector))
    print(f"Decrypted vector: {dec_vector.decrypt(local_context.secret_key())}")


@click.group()
@click.version_option(version=__version__)
def main() -> None:
    """Chemxor console."""
    pass


main.add_command(partnet)
main.add_command(serve)
main.add_command(query)
