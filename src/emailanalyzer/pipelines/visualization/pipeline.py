"""
This is a boilerplate pipeline 'visualization'
generated using Kedro 1.2.0
"""

from kedro.pipeline import Node, Pipeline  # noqa
from .nodes import generate_spam_vs_ham_pie_chart, create_evaluation_summary_data


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=create_evaluation_summary_data,
                inputs=["regressor"],
                outputs="evaluation_results",
                name="evaluate_model_node",
            ),
            Node(
                func=generate_spam_vs_ham_pie_chart,
                inputs="evaluation_results",
                outputs="email_summary_plot",
                name="plot_email_summary_node",
            ),
        ]
    )
