"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from chemxor.pipelines.cryptic_sage_default.pipeline import (
    cryptic_sage_default_pipeline,
)
from chemxor.pipelines.hello_world.pipeline import hello_pipeline
from chemxor.pipelines.iris.pipelines import iris_split_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    return {
        "cryptic_sage_default": cryptic_sage_default_pipeline,
        "iris_split": iris_split_pipeline,
        "__default__": hello_pipeline + iris_split_pipeline,
    }
