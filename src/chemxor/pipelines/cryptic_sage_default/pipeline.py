"Cryptic Sage default pipeline."

from kedro.pipeline import Pipeline

from chemxor.pipelines.cryptic_sage_default.nodes import (
    prepare_dataset_node,
    train_model_node,
)

# Assemble nodes into a pipeline
cryptic_sage_default_pipeline = Pipeline([prepare_dataset_node, train_model_node])
