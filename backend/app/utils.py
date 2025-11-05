from __future__ import annotations

import re
import string
from typing import Iterable

import requests
from bs4 import BeautifulSoup
from requests.exceptions import RequestException

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Pre-compute translation table to strip punctuation quickly.
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_STOPWORDS: Iterable[str] = ENGLISH_STOP_WORDS


def clean_text(text: str) -> str:
    """Lowercase, remove punctuation/stopwords, and normalise whitespace."""
    if not isinstance(text, str):
        return ""
    lowered = text.lower().translate(_PUNCT_TABLE)
    tokens = [token for token in lowered.split() if token and token not in _STOPWORDS]
    normalised = re.sub(r"\s+", " ", " ".join(tokens)).strip()
    return normalised


def fetch_article_text(url: str, timeout: int = 8) -> str:
    """Fetch HTML at *url* and extract readable article text."""
    if not isinstance(url, str) or not url.strip():
        return ""
    url = url.strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        return ""

    try:
        response = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
    except RequestException:
        return ""

    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
    extracted = " ".join(paragraphs)
    return extracted.strip()
