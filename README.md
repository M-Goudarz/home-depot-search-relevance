# Home Depot Search Relevance

A reproducible machine learning project for predicting the relevance of a product to a user's search query using the Home Depot Product Search Relevance dataset.

The project combines lexical matching, TF-IDF similarity, and Sentence-BERT semantic similarity, then trains regression models to predict graded relevance scores from 1.0 to 3.0.

## Project Overview

Given a search query and a candidate product, the system predicts a relevance score:

- `1.0` = irrelevant
- `2.0` = partially relevant
- `3.0` = perfect match

The final implementation is a modular Python package with data validation, preprocessing, feature engineering, training, evaluation, model serialization, batch inference, and automated tests.

## Problem Definition

Search relevance is treated as a supervised regression problem over query-product pairs. The dataset contains multiple candidate products for each query, so ranking quality matters in addition to pointwise error.

## Dataset

Source: Kaggle Home Depot Product Search Relevance competition.

The raw CSV files are not committed. See `data/README.md` for acquisition instructions.

Dataset sizes:

- Train: 74,067 rows, 5 columns
- Test: 166,693 rows, 4 columns
- Product descriptions: 124,428 rows
- Attributes: 2,044,803 rows

## Architecture

```text
raw CSVs
  -> data loading and validation
  -> merge train/test with descriptions and brand attributes
  -> canonical text preprocessing
  -> lexical + TF-IDF + Sentence-BERT features
  -> group-aware train/validation split
  -> Random Forest and LightGBM training
  -> evaluation (RMSE, MAE, NDCG@5, NDCG@10)
  -> serialized model artifact
  -> batch inference / CLI prediction
```

## Feature Engineering

The model uses 13 features:

1. `search_len` - number of stemmed query tokens
2. `word_match_title` - query/product-title token overlap
3. `word_match_desc` - query/product-description token overlap
4. `word_match_brand` - query/brand token overlap
5. `ratio_title` - title overlap normalized by query length
6. `ratio_desc` - description overlap normalized by query length
7. `ratio_brand` - brand overlap normalized by query length
8. `jaccard_title` - Jaccard similarity between query and title
9. `jaccard_desc` - Jaccard similarity between query and description
10. `jaccard_brand` - Jaccard similarity between query and brand
11. `tfidf_cos_title` - TF-IDF cosine similarity between query and title
12. `tfidf_cos_desc` - TF-IDF cosine similarity between query and description
13. `bert_semantic_title_sim` - Sentence-BERT cosine similarity between query and title

## Models

- Mean relevance baseline
- Random Forest Regressor
- LightGBM Regressor

LightGBM is the primary model and is serialized for inference.

## Evaluation Methodology

- Pointwise metrics: RMSE and MAE
- Ranking metrics: NDCG@5 and NDCG@10, computed per query and averaged
- Validation split: `GroupShuffleSplit` by `search_term`, so all candidate products for a query remain in the same partition
- Cross-validation: `GroupKFold` by `search_term`, with TF-IDF refit inside each training fold to avoid leakage

## Reported Results

The original notebook reported:

- Random Forest validation RMSE: `0.4743`
- LightGBM validation RMSE: `0.4629`

These values are preserved from the original project and have not yet been independently reproduced by the refactored pipeline. The refactored training pipeline is implemented in `src/hd_search_relevance/cli.py` and can be run with the commands below.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Data Setup

Place the Kaggle CSV files in `data/`. See `data/README.md`.

## Training

```bash
python scripts/train.py
```

Optional flags:

```bash
python scripts/train.py --skip-cv
python scripts/train.py --data-dir data --artifact-path artifacts/search_relevance_model.joblib --output-dir outputs
```

The training command writes:

- `artifacts/search_relevance_model.joblib`
- `outputs/submission.csv`
- `outputs/metrics.json`

## Inference

```bash
python scripts/predict.py --input data/test.csv --artifact artifacts/search_relevance_model.joblib --output outputs/predictions.csv
```

## Testing

```bash
pytest
```

Tests use small synthetic fixtures and do not require downloading the full dataset or model weights. The embedding integration test is skipped automatically when `local_mini_bert_model/` is unavailable.

## Limitations

- The full refactored training pipeline has not been executed end-to-end in this environment; metrics are therefore not yet reproduced.
- TF-IDF and BERT features are computed at inference time; BERT inference can be slow on CPU.
- The modular desktop GUI from the original project was not carried forward because it was a non-functional placeholder.
- The dataset and pretrained weights are not included in the repository.

## Future Improvements

- Execute and record a full end-to-end training run
- Add calibrated prediction intervals or ranking-specific models
- Optimize embedding inference with batching and caching
- Add experiment tracking and model versioning
- Add a small optional demonstration notebook

## License

The project code is released under the MIT License. Third-party data and pretrained model weights remain subject to their own terms.
