# Home Depot Search Relevance

A machine learning project for predicting the relevance of candidate products to user search queries using the [Home Depot Product Search Relevance dataset](https://www.kaggle.com/c/home-depot-product-search-relevance).

The project combines lexical matching, TF-IDF similarity, and Sentence-BERT semantic similarity with regression models to predict graded relevance scores from 1.0 to 3.0.

## Project Overview

Given a search query and a candidate product, the system predicts a relevance score:

- `1.0` — Irrelevant
- `2.0` — Partially relevant
- `3.0` — Highly relevant

The project is implemented as a modular Python package covering data validation, preprocessing, feature engineering, model training, evaluation, model serialization, batch inference, and automated testing.

It also includes a Streamlit interface for scoring and ranking candidate products.

## Problem Definition

Search relevance is formulated as a supervised regression problem over query-product pairs.

Since each search query can have multiple candidate products, model performance is evaluated using both pointwise regression metrics and ranking metrics.

## Dataset

**Source:** Kaggle — Home Depot Product Search Relevance competition.

The raw CSV files are not included in this repository. See [`data/README.md`](data/README.md) for dataset acquisition and setup instructions.

| Dataset | Rows | Columns |
|---|---:|---:|
| Train | 74,067 | 5 |
| Test | 166,693 | 4 |
| Product descriptions | 124,428 | — |
| Product attributes | 2,044,803 | — |

## Architecture

```text
Raw CSV files
    |
    v
Data loading and validation
    |
    v
Merge product descriptions and brand attributes
    |
    v
Text preprocessing
    |
    v
Feature engineering
    |-- Lexical matching
    |-- TF-IDF similarity
    |-- Sentence-BERT embeddings
    |
    v
Group-aware train/validation split
    |
    v
Model training
    |-- Random Forest
    |-- LightGBM
    |
    v
Evaluation
    |-- RMSE
    |-- MAE
    |-- NDCG@5
    |-- NDCG@10
    |
    v
Final model artifact
    |
    v
Batch inference / Streamlit interface
```

## Feature Engineering

The model uses 13 features:

| Feature | Description |
|---|---|
| `search_len` | Number of stemmed query tokens |
| `word_match_title` | Query/product-title token overlap |
| `word_match_desc` | Query/product-description token overlap |
| `word_match_brand` | Query/brand token overlap |
| `ratio_title` | Title overlap normalized by query length |
| `ratio_desc` | Description overlap normalized by query length |
| `ratio_brand` | Brand overlap normalized by query length |
| `jaccard_title` | Jaccard similarity between query and title |
| `jaccard_desc` | Jaccard similarity between query and description |
| `jaccard_brand` | Jaccard similarity between query and brand |
| `tfidf_cos_title` | TF-IDF cosine similarity between query and title |
| `tfidf_cos_desc` | TF-IDF cosine similarity between query and description |
| `bert_semantic_title_sim` | Sentence-BERT cosine similarity between query and title |

## Models

The following models are evaluated:

- Mean relevance baseline
- Random Forest Regressor
- LightGBM Regressor

LightGBM is selected as the final model and retrained on the full training dataset before being saved as a model artifact.

## Evaluation Methodology

- **Regression metrics:** RMSE and MAE.
- **Ranking metrics:** NDCG@5 and NDCG@10, computed per query and averaged.
- **Validation split:** `GroupShuffleSplit` by `search_term`, keeping all candidate products for a query in the same partition.
- **Cross-validation:** `GroupKFold` by `search_term`, with TF-IDF fitted within each training fold to avoid data leakage.

## Results

The following results were obtained from the refactored training pipeline.

### Validation results

| Model | RMSE | MAE | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|
| Mean baseline | 0.5431 | 0.4465 | 0.8658 | 0.9205 |
| Random Forest | 0.4750 | 0.3828 | 0.8888 | 0.9332 |
| LightGBM | **0.4718** | **0.3811** | **0.8903** | **0.9336** |

### Group-aware cross-validation results

| Model | Mean RMSE | Mean MAE | Mean NDCG@5 | Mean NDCG@10 |
|---|---:|---:|---:|---:|
| Random Forest | 0.4707 | 0.3792 | 0.8874 | 0.9333 |
| LightGBM | **0.4682** | **0.3779** | **0.8878** | **0.9334** |

LightGBM achieved the best overall results among the evaluated models and was selected for final training.

NDCG measures ranking quality; it should not be interpreted as classification accuracy.

## Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

Create and activate a virtual environment.

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Setup

Download the required CSV files from the Kaggle competition and place them in the `data/` directory.

Refer to [`data/README.md`](data/README.md) for the expected files and directory layout.

## Training

To run the training pipeline:

```bash
python scripts/train.py
```

Optional arguments:

```bash
python scripts/train.py --skip-cv
python scripts/train.py --data-dir data --artifact-path artifacts/search_relevance_model.joblib --output-dir outputs
```

### Resumable Feature Preparation

Sentence-BERT feature generation can take considerable time, particularly on CPU.

The project supports resumable feature preparation:

```bash
python scripts/prepare_base_features.py --split train
python scripts/prepare_base_features.py --split test
```

Generate BERT features in time-budgeted chunks:

```bash
python scripts/compute_features.py --split train
python scripts/compute_features.py --split test
```

Run each command repeatedly until the corresponding feature generation is complete.

Progress is saved in:

```text
outputs/train_features.parquet
outputs/test_features.parquet
```

Once both cached feature files are complete, train without recomputing the embeddings:

```bash
python scripts/train.py --use-cached-features
```

The training pipeline generates:

- `artifacts/search_relevance_model.joblib`
- `outputs/submission.csv`
- `outputs/metrics.json`

## Inference

Run batch predictions using the saved model:

```bash
python scripts/predict.py --input path/to/pairs.csv --artifact artifacts/search_relevance_model.joblib --output outputs/predictions.csv
```

The input CSV must contain:

- `product_title`
- `search_term`

Optional columns:

- `product_description`
- `product_brand`

Missing descriptions default to empty values, and missing brands default to `Unknown`.

## Streamlit Demo

After training and generating the model artifact, launch the interface:

```bash
streamlit run app/streamlit_app.py
```

The application:

1. Accepts a search query.
2. Provides an editable table of candidate products, including title, description, and brand.
3. Scores candidates using the saved model and inference pipeline.
4. Displays predicted relevance scores and ranked results.

The interface uses the trained model artifact rather than a separate mock prediction logic.

## Testing

Run the automated test suite:

```bash
pytest
```

Tests use small synthetic fixtures and do not require downloading the full competition dataset or model weights.

The embedding integration test is skipped automatically when `local_mini_bert_model/` is unavailable.

## Limitations

- The application ranks supplied candidate products; it does not implement a full product retrieval or search engine.
- Sentence-BERT inference can be computationally expensive on CPU.
- The project predicts relevance scores but does not include a separate mechanism for detecting invalid or meaningless search queries.
- Raw competition data, generated feature caches, model artifacts, and pretrained model weights are not included in the repository.

## Future Improvements

- Conduct ablation studies to measure the contribution of Sentence-BERT features.
- Explore learning-to-rank approaches.
- Optimize embedding inference through batching and caching.
- Add experiment tracking and model versioning.
- Improve handling of invalid or out-of-domain search queries.

## License

The project code is released under the MIT License.

Third-party datasets and pretrained model weights remain subject to their respective licenses and terms of use.