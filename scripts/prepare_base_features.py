"""Prepare and persist base text/lexical features for train and test."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hd_search_relevance.data import build_test_frame, build_training_frame, load_raw_data
from hd_search_relevance.features import add_clean_columns, add_lexical_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], required=True)
    args = parser.parse_args()
    raw = load_raw_data(Path("data"))
    builder = build_training_frame if args.split == "train" else build_test_frame
    print(f"preparing {args.split}", flush=True)
    frame = builder(raw)
    frame = add_clean_columns(frame)
    frame = add_lexical_features(frame)
    frame["bert_semantic_title_sim"] = np.nan
    Path("outputs").mkdir(exist_ok=True)
    frame.to_parquet(f"outputs/{args.split}_features.parquet", index=False)
    print(f"saved {args.split}: {frame.shape}", flush=True)


if __name__ == "__main__":
    main()
