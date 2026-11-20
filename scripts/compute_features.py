"""Compute Sentence-BERT semantic features in resumable chunks."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hd_search_relevance.config import AppConfig
from hd_search_relevance.embeddings import encode_texts, load_embedding_model, rowwise_cosine

CHUNK_SIZE = 5000
TIME_BUDGET_SECONDS = 480


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], required=True)
    args = parser.parse_args()

    path = Path(f"outputs/{args.split}_features.parquet")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run scripts/prepare_base_features.py --split {args.split} first."
        )

    frame = pd.read_parquet(path)
    missing = np.flatnonzero(frame["bert_semantic_title_sim"].isna().to_numpy())
    if missing.size == 0:
        print(f"{args.split}: all BERT features already computed ({len(frame)} rows)")
        return

    local_model = Path("local_mini_bert_model")
    model_name = str(local_model) if local_model.exists() else AppConfig().model.embedding_model_name
    print(f"{args.split}: loading embedding model {model_name}", flush=True)
    model = load_embedding_model(model_name)

    start = time.time()
    processed = 0
    while time.time() - start < TIME_BUDGET_SECONDS:
        missing = np.flatnonzero(frame["bert_semantic_title_sim"].isna().to_numpy())
        if missing.size == 0:
            break
        chunk = missing[:CHUNK_SIZE]
        search_embeddings = encode_texts(
            model, frame.loc[chunk, "clean_search_term"].tolist(), batch_size=128
        )
        title_embeddings = encode_texts(
            model, frame.loc[chunk, "clean_product_title"].tolist(), batch_size=128
        )
        frame.loc[chunk, "bert_semantic_title_sim"] = rowwise_cosine(
            search_embeddings, title_embeddings
        ).astype(float)
        frame.to_parquet(path, index=False)
        processed += len(chunk)
        remaining = int(frame["bert_semantic_title_sim"].isna().sum())
        print(
            f"{args.split}: processed {processed} this run; remaining {remaining}; "
            f"elapsed {time.time() - start:.0f}s",
            flush=True,
        )

    remaining = int(frame["bert_semantic_title_sim"].isna().sum())
    print(f"{args.split}: done for now; remaining {remaining}", flush=True)


if __name__ == "__main__":
    main()
