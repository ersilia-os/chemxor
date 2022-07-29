"""Pipelines."""

from kedro.pipeline import Pipeline

from chemxor.pipelines.nodes import (
    init_convnet_model_node,
    init_smiles_data_module_node,
    train_convnet_model_node,
)

# Assemble nodes into a pipeline
convnet_linear_one_pipeline = Pipeline(
    [
        init_smiles_data_module_node,
        init_convnet_model_node,
        train_convnet_model_node,
    ]
)
