"Convnet default pipeline."

from kedro.pipeline import Pipeline

from chemxor.pipelines.convnet_default.nodes import (
    init_convnet_model_node,
    init_mnist_data_module_node,
    train_convnet_model_node,
)

# Assemble nodes into a pipeline
convnet_default_pipeline = Pipeline(
    [
        init_mnist_data_module_node,
        init_convnet_model_node,
        train_convnet_model_node,
    ]
)
