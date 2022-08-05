"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from chemxor.pipelines.olinda_train_classification import olinda_one_cls_pipeline
from chemxor.pipelines.olinda_train_regression import olinda_one_reg_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    return {
        "olinda_one_reg": olinda_one_reg_pipeline,
        "olinda_one_cls": olinda_one_cls_pipeline,
        "__default__": olinda_one_reg_pipeline,
    }
