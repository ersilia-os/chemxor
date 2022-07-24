"""Convert smiles to imgs pipeline."""

from kedro.pipeline import Pipeline

from chemxor.pipelines.sm_to_img.nodes import convert_smiles_to_imgs_node

# Assemble nodes into a pipeline
convert_smiles_to_imgs_pipeline = Pipeline([convert_smiles_to_imgs_node])
