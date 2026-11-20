from pathlib import Path

import numpy as np
import pytest

from hd_search_relevance.embeddings import encode_texts, load_embedding_model

LOCAL_MODEL = Path("local_mini_bert_model")


@pytest.mark.skipif(not LOCAL_MODEL.exists(), reason="Local Sentence Transformer model is not available")
def test_embedding_model_small_input():
    model = load_embedding_model(str(LOCAL_MODEL))
    texts = ["drill bit", "hammer drill", ""]
    embeddings = encode_texts(model, texts, batch_size=2, normalize_embeddings=True)
    assert embeddings.shape == (3, 384)
    assert embeddings.dtype == np.float32
    assert np.all(np.isfinite(embeddings))
    # Normalized embeddings should have unit length for non-empty inputs.
    norms = np.linalg.norm(embeddings[:2], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
