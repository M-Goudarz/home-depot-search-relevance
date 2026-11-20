import pandas as pd

from hd_search_relevance.data import build_training_frame, extract_brands


def test_extract_brands_and_merge_integrity():
    attributes = pd.DataFrame(
        {
            "product_uid": [1, 1, 2],
            "name": ["MFG Brand Name", "Color", "MFG Brand Name"],
            "value": ["DeWalt", "Yellow", "Makita"],
        }
    )
    brands = extract_brands(attributes)
    assert list(brands.columns) == ["product_uid", "product_brand"]
    assert brands["product_uid"].is_unique

    train = pd.DataFrame(
        {
            "id": [1, 2],
            "product_uid": [1, 2],
            "product_title": ["drill", "saw"],
            "search_term": ["drill", "saw"],
            "relevance": [3.0, 2.0],
        }
    )
    descriptions = pd.DataFrame(
        {
            "product_uid": [1, 2],
            "product_description": ["a drill", "a saw"],
        }
    )
    raw = {
        "train": train,
        "descriptions": descriptions,
        "attributes": attributes,
    }
    merged = build_training_frame(raw)
    assert len(merged) == len(train)
    assert merged["id"].tolist() == [1, 2]
    assert merged.loc[0, "product_brand"] == "DeWalt"
    assert merged.loc[1, "product_brand"] == "Makita"
    assert merged["product_description"].isna().sum() == 0


def test_missing_brand_filled():
    attributes = pd.DataFrame(
        {"product_uid": [1], "name": ["Color"], "value": ["Yellow"]}
    )
    train = pd.DataFrame(
        {
            "id": [1],
            "product_uid": [1],
            "product_title": ["drill"],
            "search_term": ["drill"],
            "relevance": [3.0],
        }
    )
    descriptions = pd.DataFrame({"product_uid": [1], "product_description": ["drill"]})
    raw = {"train": train, "descriptions": descriptions, "attributes": attributes}
    merged = build_training_frame(raw)
    assert merged.loc[0, "product_brand"] == "Unknown"
