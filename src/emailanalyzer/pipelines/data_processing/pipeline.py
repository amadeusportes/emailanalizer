"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 1.2.0
"""

from kedro.pipeline import Node, Pipeline  # noqa
from .nodes import preprocess_emails


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=preprocess_emails,
                inputs="emails",
                outputs="preprocessed_emails",
                name="preprocess_emails_node",
            )
        ]
    )
