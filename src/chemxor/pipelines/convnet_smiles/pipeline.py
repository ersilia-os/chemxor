"Convnet smiles pipeline."

from kedro.pipeline import Pipeline

from chemxor.pipelines.convnet_smiles.nodes import (
    init_convnet_model_node,
    init_smiles_data_module_node,
    train_convnet_model_node,
)

# Assemble nodes into a pipeline
convnet_smiles_pipeline = Pipeline(
    [
        init_smiles_data_module_node,
        init_convnet_model_node,
        train_convnet_model_node,
    ]
)
