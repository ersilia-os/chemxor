"""Iris pipelines."""

from kedro.pipeline import Pipeline, node

from chemxor.pipelines.iris.nodes import split_data


iris_split_pipeline = Pipeline(
    [
        node(
            split_data,
            ["iris_data", "params:example_test_data_ratio"],
            dict(
                train_x="example_train_x",
                train_y="example_train_y",
                test_x="example_test_x",
                test_y="example_test_y",
            ),
            name="split",
        )
    ]
)
