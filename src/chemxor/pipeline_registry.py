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
from chemxor.pipelines.cryptic_sage_default.pipeline import (
    cryptic_sage_default_pipeline,
)
from chemxor.pipelines.hello_world.pipeline import hello_pipeline
from chemxor.pipelines.iris.pipelines import iris_split_pipeline
from chemxor.pipelines.sm_to_img.pipeline import convert_smiles_to_imgs_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    return {
        "convnet_linear_one": convnet_linear_one_pipeline,
        "convnet": convnet_default_pipeline,
        "convnet-smiles": convnet_smiles_pipeline,
        "cryptic_sage_default": cryptic_sage_default_pipeline,
        "sm_to_imgs": convert_smiles_to_imgs_pipeline,
        "iris_split": iris_split_pipeline,
        "__default__": hello_pipeline + iris_split_pipeline,
    }
