"""Partitioned network Service."""

from typing import Optional, Union

from flask import Blueprint, Flask
from pytorch_lightning import LightningModule
from torch import nn

from chemxor.service.blocs import generate_blueprint


class PartitionNetServer:
    """Partitioned network service."""

    def __init__(
        self: "PartitionNetServer",
        part_net: Optional[Union[nn.Module, LightningModule]] = None,
        flask_app: Optional[Flask] = None,
    ) -> None:
        """Init."""
        self.part_net = part_net
        self.flask_app = flask_app or Flask(str(part_net))
        self.flask_bp = generate_blueprint(self.part_net)
        self.flask_app.register_blueprint(self.flask_bp)

    def get_flask_app(self: "PartitionNetServer") -> Optional[Flask]:
        """Return Flask app if present."""
        return self.flask_app

    def get_flask_blueprint(self: "PartitionNetServer") -> Optional[Blueprint]:
        """Return Flask blueprint if present."""
        return self.flask_bp

    def run(self: "PartitionNetServer") -> None:
        """Start flask server."""
        self.flask_app.run()
