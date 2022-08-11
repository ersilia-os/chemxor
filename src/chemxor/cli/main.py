"""Chemxor CLI."""

from pathlib import Path
from typing import Optional

import click

from chemxor.model import (
    FHEOlindaNet,
    FHEOlindaNetOne,
    FHEOlindaNetZero,
    OlindaNet,
    OlindaNetOne,
    OlindaNetZero,
)
from chemxor.service.part_net_client import PartitionNetClient
from chemxor.service.part_net_server import PartitionNetServer
from .. import __version__


@click.command()
@click.argument("model", type=click.Choice(["olinda", "olinda_zero", "olinda_one"]))
@click.option("-cp", "--checkpoint", type=click.Path(exists=True))
@click.option("-o", "--output", default=1)
def serve(model: str, checkpoint: Optional[Path], output: int) -> None:
    """Serve models."""
    if model == "olinda":
        model = OlindaNet(output=output)
        if checkpoint is not None:
            model.load(checkpoint)
        fhe_model = FHEOlindaNet(model)
    elif model == "olinda_zero":
        model = OlindaNetZero(output=output)
        if checkpoint is not None:
            model.load(checkpoint)
        fhe_model = FHEOlindaNetZero(model)
    if model == "olinda_one":
        model = OlindaNetOne(output=output)
        if checkpoint is not None:
            model.load(checkpoint)
        fhe_model = FHEOlindaNetOne(model)
    server = PartitionNetServer(fhe_model)
    server.run()


@click.command()
@click.argument("url")
@click.argument("smile")
def query(url: str, smile: str) -> None:
    """Query models."""
    client = PartitionNetClient(url)
    dec_out = client.query(smile)
    print(f"Model output: {dec_out}")


@click.group()
@click.version_option(version=__version__)
def main() -> None:
    """Chemxor console."""
    pass


main.add_command(serve)
main.add_command(query)
