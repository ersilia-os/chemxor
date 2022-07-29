"""PartNet CLI commands."""

import click


@click.command()
def serve() -> None:
    """Serve PartNet models."""
    pass


@click.command()
def query() -> None:
    """Query PartNet models."""
    pass


@click.group()
def partnet() -> None:
    """Partioned FHENetwork commands."""
    pass


partnet.add_command(serve)
partnet.add_command(query)
