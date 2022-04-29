"Cryptic Sage default pipeline."

from kedro.pipeline import Pipeline

from chemxor.pipelines.cryptic_sage_default.nodes import (
    init_cryptic_sage_model_node,
    init_osm_data_module_node,
    train_cryptic_sage_model_node,
)

# Assemble nodes into a pipeline
cryptic_sage_default_pipeline = Pipeline(
    [
        init_osm_data_module_node,
        init_cryptic_sage_model_node,
        train_cryptic_sage_model_node,
    ]
)
