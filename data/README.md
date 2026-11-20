# Data

This directory is intentionally excluded from Git.

The project uses the Home Depot Product Search Relevance Kaggle competition data:

- `train.csv`
- `test.csv`
- `product_descriptions.csv`
- `attributes.csv`
- `sample_submission.csv`

Download the files from the Kaggle competition page and place them in this directory. Do not redistribute the data without permission from Kaggle and the competition organizers.

Expected columns:

- `train.csv`: `id`, `product_uid`, `product_title`, `search_term`, `relevance`
- `test.csv`: `id`, `product_uid`, `product_title`, `search_term`
- `product_descriptions.csv`: `product_uid`, `product_description`
- `attributes.csv`: `product_uid`, `name`, `value`
- `sample_submission.csv`: `id`, `relevance`
