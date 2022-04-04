"""February CLI."""

from typing import Any

import click
from kedro.framework.session import KedroSession


@click.group(name="February")
def cli() -> None:
    """Kedro plugin for printing the pipeline in JSON format."""
    pass


@cli.command()
@click.pass_obj
def hello(metadata: Any) -> None:
    """Display the pipeline in JSON format."""
    print("hello from february.")
    session = KedroSession.create(metadata.package_name)
    context = session.load_context()
    print(context.pipeline.to_json())
