"""Canonical text preprocessing for Home Depot search relevance."""

from __future__ import annotations

import re

import numpy as np
from typing import Any

import pandas as pd
from nltk.stem.porter import PorterStemmer

_stemmer = PorterStemmer()

_PUNCTUATION_RE = re.compile(r"[^a-z0-9\s]")
_WHITESPACE_RE = re.compile(r"\s+")

# Multiword synonyms are applied before tokenization. Longer phrases must come
# first so that, for example, "cu ft" is not partially replaced as "cu".
_PHRASE_SYNONYMS: dict[str, str] = {
    "cu ft": "cubic foot",
    "cu in": "cubic inch",
    "sq ft": "square foot",
    "sq in": "square inch",
    "btu hr": "british thermal unit per hour",
}

_TOKEN_SYNONYMS: dict[str, str] = {
    "gal": "gallon",
    "oz": "ounce",
    "lb": "pound",
    "lbs": "pound",
    "in": "inch",
    "ft": "foot",
    "cu": "cubic",
    "sq": "square",
    "pk": "pack",
    "ea": "each",
    "w": "watt",
    "v": "volt",
    "amp": "ampere",
    "hp": "horsepower",
    "btu": "british thermal unit",
    "cfm": "cubic feet per minute",
    "psi": "pounds per square inch",
    "rpm": "revolutions per minute",
}

_UNIT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("gallons|gallon|gal", "gallon"),
    ("ounces|ounce|oz", "ounce"),
    ("pounds|pound|lb|lbs", "pound"),
    ("inches|inch|in", "inch"),
    ("feet|foot|ft", "foot"),
    ("volts|volt|v", "volt"),
    ("watts|watt|w", "watt"),
)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(result, (bool, np.bool_)):
        return bool(result)
    try:
        return bool(result)
    except (TypeError, ValueError):
        return False


def clean_text(value: Any) -> str:
    """Return a normalized, stemmed representation of free text.

    The same function is used for train, validation, and inference text. Missing
    values become empty strings. Punctuation is converted to whitespace before
    stemming so that values such as ``1-gal.`` and ``3/4 in.`` normalize
    consistently.
    """
    if _is_missing(value):
        return ""

    text = str(value).lower().strip()
    if not text:
        return ""

    # Phrase-level synonyms (longest first) before tokenization.
    for phrase, replacement in sorted(
        _PHRASE_SYNONYMS.items(), key=lambda item: len(item[0]), reverse=True
    ):
        text = re.sub(rf"\b{re.escape(phrase)}\b", replacement, text)

    # Normalize punctuation to whitespace.
    text = _PUNCTUATION_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()

    # Token-level synonyms.
    tokens = text.split()
    tokens = [_TOKEN_SYNONYMS.get(token, token) for token in tokens]
    text = " ".join(tokens)

    # Normalize common unit spellings after numeric values.
    for alternatives, canonical in _UNIT_PATTERNS:
        text = re.sub(
            rf"(\d+)\s*({alternatives})\b",
            rf"\1 {canonical}",
            text,
        )

    # Normalize simple dimension expressions such as "2 x 4".
    text = re.sub(r"(\d+)\s*x\s*(\d+)", r"\1 x \2", text)

    text = _WHITESPACE_RE.sub(" ", text).strip()
    stemmed = [_stemmer.stem(token) for token in text.split()]
    return " ".join(stemmed)

