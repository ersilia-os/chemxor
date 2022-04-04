"""Hello world pipeline."""

from kedro.pipeline import Pipeline

from chemxor.pipelines.hello_world.nodes import (
    join_statements_node,
    return_greeting_node,
)

# Assemble nodes into a pipeline
hello_pipeline = Pipeline([return_greeting_node, join_statements_node])
