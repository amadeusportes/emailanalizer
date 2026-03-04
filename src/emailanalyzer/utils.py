import logging
import re

logger = logging.getLogger(__name__)


def title(title: str):
    logger.info(f"\n{'═'*60}\n  {title}\n{'═'*60}")


def section(title: str):
    logger.info(f"\n{'*'*20} {title} {'*'*20}")


def _normalize_and_mask_text(text: str) -> str:
    """
    Cleans the input text by performing the following operations:
    1. Converts text to lower case.
    2. Replaces URLs with the token 'url'.
    3. Replaces currency symbols (£, $, €) with the token 'money'.
    4. Replaces isolated numbers with the token 'num'.
    5. Removes all remaining punctuation/non-word characters.
    6. Strips and normalizes extra whitespace.

    Args:
        text (str): The raw text string.

    Returns:
        str: The cleaned text string.
    """
    # 1. Convert to lower case
    text = text.lower()

    # 2. Mask URLs
    text = re.sub(r"http\S+|www\S+", " url ", text)

    # 3. Mask currency symbols (£, $, €)
    text = re.sub(r"\£|\$|\€", " money ", text)

    # 4. Mask isolated numbers
    text = re.sub(r"\b\d+\b", " num ", text)

    # 5. Remove all remaining punctuation/non-word characters (keep only alphanumeric and spaces)
    text = re.sub(r"[^\w\s]", " ", text)

    # 6. Strip and normalize extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text
