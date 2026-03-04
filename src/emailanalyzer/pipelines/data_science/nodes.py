"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 1.2.0
"""

import typing as t
import pandas as pd
import logging
from emailanalyzer.utils import section, title

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)

models = {
    "Naive Bayes": MultinomialNB(alpha=0.1),
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2),  # words + word pairs
    stop_words="english",
    sublinear_tf=True,
)


def _calculate_metrics(name: str, model, X_test, y_test) -> dict:
    """
    Calculates performance metrics for a given trained model.

    Args:
        name (str): The name of the model being evaluated.
        model: The trained scikit-learn model instance.
        X_test: The features for the test dataset.
        y_test: The true labels for the test dataset.

    Returns:
        dict: A dictionary containing the model name, accuracy, precision, recall, and F1 score.
    """
    y_pred = model.predict(X_test)
    return {
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, pos_label="spam"), 4),
        "Recall": round(recall_score(y_test, y_pred, pos_label="spam"), 4),
        "F1 Score": round(f1_score(y_test, y_pred, pos_label="spam"), 4),
    }


def vectorize_and_split_data(df: pd.DataFrame) -> t.Tuple:
    """
    Transforms the cleaned text into TF-IDF numerical features and splits the dataset
    into training and testing sets.

    Args:
        df (pd.DataFrame): The preprocessed DataFrame containing 'clean_text' and 'label' columns.

    Returns:
        tuple: (X_train, X_test, y_train, y_test) data splits for model training and evaluation.
    """
    title("Split data")

    # Transform text data into TF-IDF feature vectors
    X = vectorizer.fit_transform(df["clean_text"])

    # Extract target labels
    y = df["label"]

    # Split data chronologically or randomly as needed (using stratified sampling here)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}")
    logger.info(f"y_train shape: {y_train.shape}")
    logger.info(f"y_test shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test


def train_and_select_best_model(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> dict:
    """
    Trains multiple machine learning models and selects the best one based on F1 Score.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training labels.
        y_test (pd.Series): Testing labels.

    Returns:
        dict: A dictionary containing the best trained model, test sets, and the vectorizer.
    """
    title("Training models")
    trained, results = {}, []

    # Iterate through all configured models
    for name, model in models.items():
        # Train the model on the training set
        model.fit(X_train, y_train)
        trained[name] = model

        # Calculate performance metrics on the test set
        metrics = _calculate_metrics(name, model, X_test, y_test)
        results.append(metrics)
        logger.info(f"Model trained: {name}")

    # Compare all models' performance
    results_df = pd.DataFrame(results).set_index("Model")
    logger.info("MODEL COMPARISON")
    logger.info(results_df)

    # Select the model with the highest F1 Score
    best_name = results_df["F1 Score"].idxmax()
    best_model = trained[best_name]

    logger.info(f"Best model: {best_name}")
    logger.info(f"Best model metrics: {results_df.loc[best_name]}")
    logger.info(f"Best model accuracy: {results_df.loc[best_name]['Accuracy']}")
    logger.info(f"Best model precision: {results_df.loc[best_name]['Precision']}")
    logger.info(f"Best model recall: {results_df.loc[best_name]['Recall']}")
    logger.info(f"Best model F1 score: {results_df.loc[best_name]['F1 Score']}")

    # Package the best model, test data, and vectorizer for downstream nodes
    return {
        "model": best_model,
        "x_test": X_test,
        "y_test": y_test,
        "vectorizer": vectorizer,
    }
