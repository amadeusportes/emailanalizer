"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 1.2.0
"""

from kedro.pipeline import Node, Pipeline  # noqa
from .nodes import vectorize_and_split_data, train_and_select_best_model


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=vectorize_and_split_data,
                inputs="preprocessed_emails",
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            Node(
                func=train_and_select_best_model,
                inputs=["X_train", "X_test", "y_train", "y_test"],
                outputs="regressor",
                name="train_model_node",
            ),
        ]
    )
