import math

import numpy as np
import pandas as pd
import pytest

from hd_search_relevance.preprocessing import clean_text


def test_clean_text_empty_and_missing():
    assert clean_text(None) == ""
    assert clean_text("") == ""
    assert clean_text(float("nan")) == ""
    assert clean_text(pd.NA) == ""


def test_clean_text_punctuation_and_whitespace():
    assert clean_text("  Drill,   Bit!!!  ") == clean_text("drill bit")
    assert "  " not in clean_text("a   b")


def test_clean_text_synonyms_phrase():
    result = clean_text("2 cu ft pipe")
    assert "cubic" in result
    assert "foot" in result


def test_clean_text_units():
    result = clean_text("5 gal bucket")
    assert "5 gallon" in result


def test_clean_text_repeated_words():
    result = clean_text("drill drill bit")
    assert result.count("drill") == 2


def test_clean_text_numbers_and_brand():
    result = clean_text("DeWalt 20V MAX Drill")
    assert result != ""
    assert "dewalt" in result or "dewalt" in result.lower()
