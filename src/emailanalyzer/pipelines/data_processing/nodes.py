"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 1.2.0
"""

import pandas as pd
import logging

from emailanalyzer.utils import section, title
from emailanalyzer.utils import _normalize_and_mask_text


logger = logging.getLogger(__name__)


def _extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts numerical features from the raw text column.

    Args:
        df: Input DataFrame containing a 'text' column.

    Returns:
        DataFrame with additional feature columns (word_count, exclamation_count, has_url, has_money, has_phone)
    """
    # Count the total number of words in the email
    df["word_count"] = df["text"].str.split().str.len()

    # Count the number of exclamation marks
    df["exclamation_count"] = df["text"].str.count(r"!")

    # Flag to indicate if the email contains common URL patterns or TLDs
    df["has_url"] = (
        df["text"]
        .str.contains(r"http|www|\.com|\.co\.uk", case=False, regex=True)
        .astype(int)
    )

    # Flag to indicate if the email contains currency symbols or spammy money-related keywords
    df["has_money"] = (
        df["text"]
        .str.contains(
            r"\£|\$|\€|free|win|cash|prize|earn|claim|awarded", case=False, regex=True
        )
        .astype(int)
    )

    # Flag to indicate if the email contains a phone number (10-13 digits)
    df["has_phone"] = (
        df["text"]
        .str.contains(r"\b0\d{9,10}\b|\b\+\d{10,13}\b", regex=True)
        .astype(int)
    )
    return df


def preprocess_emails(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main preprocessing pipeline node.
    1. Extracts numerical features from the text.
    2. Cleans the text column to create a normalized 'clean_text' column.
    3. Drops null values and duplicate emails.

    Args:
        df: Raw input DataFrame

    Returns:
        Processed DataFrame ready for model training.
    """
    title("Main preprocessing pipeline node")
    section("Raw data")
    logger.info(df["text"])
    df = _extract_features(df)
    df["clean_text"] = df["text"].apply(_normalize_and_mask_text)
    section("Cleaned data")
    logger.info(df["clean_text"])
    return df.dropna().drop_duplicates(subset=["text"])
