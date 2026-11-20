"""Data loading and integration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

TRAIN_COLUMNS = ["id", "product_uid", "product_title", "search_term", "relevance"]
TEST_COLUMNS = ["id", "product_uid", "product_title", "search_term"]
DESCRIPTION_COLUMNS = ["product_uid", "product_description"]
ATTRIBUTE_COLUMNS = ["product_uid", "name", "value"]


class DataValidationError(ValueError):
    """Raised when input data does not satisfy expected invariants."""


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required data file not found: {path}")


def _require_columns(frame: pd.DataFrame, required: list[str], frame_name: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DataValidationError(
            f"{frame_name} is missing required columns: {missing}"
        )


def load_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    _require_file(path)
    try:
        return pd.read_csv(path, encoding="ISO-8859-1", **kwargs)
    except Exception as exc:  # pragma: no cover - defensive
        raise DataValidationError(f"Could not read CSV {path}: {exc}") from exc


def load_raw_data(data_dir: Path | str = "data") -> dict[str, pd.DataFrame]:
    """Load and validate all raw Home Depot CSV files."""
    data_dir = Path(data_dir)
    frames = {
        "train": load_csv(data_dir / "train.csv"),
        "test": load_csv(data_dir / "test.csv"),
        "descriptions": load_csv(data_dir / "product_descriptions.csv"),
        "attributes": load_csv(data_dir / "attributes.csv"),
    }
    sample_path = data_dir / "sample_submission.csv"
    if sample_path.is_file():
        frames["sample_submission"] = load_csv(sample_path)

    _require_columns(frames["train"], TRAIN_COLUMNS, "train.csv")
    _require_columns(frames["test"], TEST_COLUMNS, "test.csv")
    _require_columns(frames["descriptions"], DESCRIPTION_COLUMNS, "product_descriptions.csv")
    _require_columns(frames["attributes"], ATTRIBUTE_COLUMNS, "attributes.csv")

    if frames["train"]["id"].duplicated().any():
        raise DataValidationError("train.csv contains duplicate id values")
    if frames["test"]["id"].duplicated().any():
        raise DataValidationError("test.csv contains duplicate id values")
    if frames["descriptions"]["product_uid"].duplicated().any():
        raise DataValidationError("product_descriptions.csv contains duplicate product_uid values")
    if frames["train"]["relevance"].isna().any():
        raise DataValidationError("train.csv contains missing relevance values")

    return frames


def extract_brands(attributes: pd.DataFrame) -> pd.DataFrame:
    """Extract one brand value per product_uid from the attributes table."""
    _require_columns(attributes, ATTRIBUTE_COLUMNS, "attributes")
    brand_rows = attributes.loc[attributes["name"] == "MFG Brand Name", ["product_uid", "value"]].copy()
    brand_rows = brand_rows.rename(columns={"value": "product_brand"})

    # Detect conflicting duplicate brands before deduplication.
    duplicates = brand_rows.duplicated(subset=["product_uid"], keep=False)
    if duplicates.any():
        conflicting = brand_rows.loc[duplicates].drop_duplicates()
        grouped = conflicting.groupby("product_uid")["product_brand"].nunique()
        if (grouped > 1).any():
            examples = grouped[grouped > 1].index.tolist()[:5]
            raise DataValidationError(
                f"Multiple conflicting brands found for product_uids: {examples}"
            )

    brand_rows = brand_rows.drop_duplicates(subset=["product_uid"], keep="first")
    return brand_rows[["product_uid", "product_brand"]]


def build_training_frame(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Integrate train, descriptions, and brand attributes without row multiplication."""
    train = raw["train"].copy()
    descriptions = raw["descriptions"].copy()
    brands = extract_brands(raw["attributes"])

    original_rows = len(train)
    original_ids = train["id"].tolist()

    merged = train.merge(descriptions, on="product_uid", how="left", validate="many_to_one")
    merged = merged.merge(brands, on="product_uid", how="left", validate="many_to_one")

    if len(merged) != original_rows:
        raise DataValidationError(
            f"Training merge changed row count from {original_rows} to {len(merged)}"
        )
    if merged["id"].tolist() != original_ids:
        raise DataValidationError("Training merge changed row order or IDs")

    merged["product_description"] = merged["product_description"].fillna("")
    merged["product_brand"] = merged["product_brand"].fillna("Unknown")
    merged["product_title"] = merged["product_title"].fillna("")
    merged["search_term"] = merged["search_term"].fillna("")

    return merged


def build_test_frame(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Integrate test, descriptions, and brand attributes while preserving order."""
    test = raw["test"].copy()
    descriptions = raw["descriptions"].copy()
    brands = extract_brands(raw["attributes"])

    original_rows = len(test)
    original_ids = test["id"].tolist()

    merged = test.merge(descriptions, on="product_uid", how="left", validate="many_to_one")
    merged = merged.merge(brands, on="product_uid", how="left", validate="many_to_one")

    if len(merged) != original_rows:
        raise DataValidationError(
            f"Test merge changed row count from {original_rows} to {len(merged)}"
        )
    if merged["id"].tolist() != original_ids:
        raise DataValidationError("Test merge changed row order or IDs")

    merged["product_description"] = merged["product_description"].fillna("")
    merged["product_brand"] = merged["product_brand"].fillna("Unknown")
    merged["product_title"] = merged["product_title"].fillna("")
    merged["search_term"] = merged["search_term"].fillna("")

    return merged
