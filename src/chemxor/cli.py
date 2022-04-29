"""Chemxor CLI."""

from typing import Any

import click
from kedro.framework.project import pipelines


@click.group(name="Chemxor")
def cli() -> None:
    """Chemxor CLI."""
    pass


@cli.command()
@click.pass_obj
def hello(metadata: Any) -> None:
    """Display the pipeline in JSON format."""
    print("hello from chemxor.")
    print(f"{[pipeline for pipeline in pipelines]}")
