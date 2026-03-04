"""
This is a boilerplate pipeline 'visualization'
generated using Kedro 1.2.0
"""

import typing as t
import pandas as pd
import logging
import matplotlib.pyplot as plt
from emailanalyzer.utils import section
from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger(__name__)


def create_evaluation_summary_data(regressor: dict) -> pd.DataFrame:
    """
    Evaluates the trained model on test data and returns a summary DataFrame
    intended for plotting. Calculates and logs the classification report
    and confusion matrix.

    Args:
        regressor (dict): A dictionary containing the trained model ('model')
            and the test sets ('x_test', 'y_test').

    Returns:
        pd.DataFrame: A two-row DataFrame summarizing correctly predicted
            spam and ham counts.
    """
    section("Evaluates the trained model on test data")

    # Run predictions on the test dataset
    y_pred = regressor["model"].predict(regressor["x_test"])

    # Calculate and log detailed classification metrics
    cl = classification_report(
        regressor["y_test"], y_pred, target_names=["ham", "spam"]
    )
    logger.info("Classification Report:")
    logger.info(cl)

    # Generate the confusion matrix for plotting counts
    cm = confusion_matrix(regressor["y_test"], y_pred, labels=["spam", "ham"])
    cm_df = pd.DataFrame(cm, index=["spam", "ham"], columns=["spam", "ham"])
    logger.info("Confusion Matrix:")
    logger.info(cm_df)

    # Extract the true positive ('spam' predicted 'spam') and true negative ('ham' predicted 'ham')
    # counts to be plotted in the pie chart. Notice 'ham' mapped falsely under 'spam' column in the original dataset is actually False Positive.
    # The original implementation maps:
    # 'spam'/'spam' -> Actual Spam detected as Spam
    # 'spam'/'ham' -> Actual Spam detected as Ham (False Negatives)
    return pd.DataFrame(
        {
            "Category": ["Spam", "Ham"],
            "Values": [cm_df["spam"]["spam"], cm_df["spam"]["ham"]],
        }
    )


def generate_spam_vs_ham_pie_chart(pd: pd.DataFrame) -> t.Any:
    """
    Generates a pie chart comparing the predicted 'Spam' vs 'Ham' email evaluation counts.

    Args:
        pd (pd.DataFrame): A DataFrame containing 'Category' and 'Values' columns.

    Returns:
        matplotlib.figure.Figure: A matplotlib figure object containing the plot,
                                  or None if plotting fails.
    """
    section("Generating email summary plot...")

    try:
        # Create a new matplotlib figure
        f = plt.figure(figsize=(10, 6))

        # Draw a pie chart using the summarized DataFrame values
        plt.pie(pd["Values"], labels=pd["Category"], colors=["green", "red"])
        plt.title("Actual Spam")

        # Adjust layout and return the rendered figure object (and close the window instance)
        plt.tight_layout()
        plt.close(f)
        return f
    except Exception as e:
        logger.error(f"Failed to plot SHAP summary: {e}")
        return None
