"""Sentence Transformer embedding helpers."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def get_device(requested: str | None = None) -> str:
    """Return a usable torch device string."""
    if requested is not None:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_embedding_model(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str | None = None,
) -> SentenceTransformer:
    """Load a Sentence Transformer model on CPU or GPU."""
    return SentenceTransformer(model_name, device=get_device(device))


def encode_texts(
    model: SentenceTransformer,
    texts: Sequence[str],
    batch_size: int = 64,
    normalize_embeddings: bool = True,
    show_progress_bar: bool = False,
) -> np.ndarray:
    """Encode a sequence of texts into a dense embedding matrix."""
    cleaned = ["" if text is None else str(text) for text in texts]
    embeddings = model.encode(
        cleaned,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=show_progress_bar,
    )
    return np.asarray(embeddings, dtype=np.float32)


def rowwise_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Compute cosine similarity for aligned rows."""
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    if left.shape != right.shape:
        raise ValueError(f"Embedding shapes differ: {left.shape} vs {right.shape}")
    if left.size == 0:
        return np.zeros(0, dtype=np.float32)
    # If embeddings are already normalized this is the cosine similarity.
    left_norm = np.linalg.norm(left, axis=1, keepdims=True)
    right_norm = np.linalg.norm(right, axis=1, keepdims=True)
    denom = np.clip(left_norm * right_norm, a_min=1e-12, a_max=None)
    return np.sum(left * right, axis=1) / denom.flatten()
