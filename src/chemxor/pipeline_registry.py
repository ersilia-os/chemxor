"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline


from chemxor.pipelines.pipelines import convnet_linear_one_pipeline
from chemxor.pipelines.convnet_default.pipeline import (
    convnet_default_pipeline,
)
from chemxor.pipelines.convnet_smiles.pipeline import (
    convnet_smiles_pipeline,
)


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    return {
        "convnet_linear_one": convnet_linear_one_pipeline,
        "convnet": convnet_default_pipeline,
        "convnet-smiles": convnet_smiles_pipeline,
        "__default__": convnet_smiles_pipeline,
    }
