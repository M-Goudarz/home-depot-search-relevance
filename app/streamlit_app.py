"""Streamlit demo for the Home Depot search relevance model."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hd_search_relevance.artifacts import load_artifact
from hd_search_relevance.embeddings import load_embedding_model
from hd_search_relevance.inference import score_candidates

st.set_page_config(
    page_title="Home Depot Search Relevance",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 Home Depot Search Relevance")
st.caption(
    "Score and rank candidate products for a search query using the trained model."
)

DEFAULT_ARTIFACT = Path("artifacts/search_relevance_model.joblib")


@st.cache_resource(show_spinner=False)
def get_artifact(path_str: str) -> dict:
    return load_artifact(path_str)


@st.cache_resource(show_spinner=False)
def get_embedding_model(model_name: str):
    return load_embedding_model(model_name)


def resolve_embedding_model_name(artifact: dict) -> str:
    local_model = Path("local_mini_bert_model")
    if local_model.exists():
        return str(local_model)
    return artifact["config"].get(
        "embedding_model_name", "sentence-transformers/all-MiniLM-L6-v2"
    )


if not DEFAULT_ARTIFACT.is_file():
    st.error(
        "Model artifact not found. Train the model first:\n\n"
        "`python scripts/train.py`\n\n"
        f"Expected artifact: `{DEFAULT_ARTIFACT}`"
    )
    st.stop()

try:
    artifact = get_artifact(str(DEFAULT_ARTIFACT))
except Exception as exc:  # pragma: no cover - UI error path
    st.error(f"Could not load the model artifact: {exc}")
    st.stop()

with st.sidebar:
    st.header("Model")
    st.write(f"Features: **{len(artifact['feature_names'])}**")
    st.write(f"Embedding: `{artifact['config'].get('embedding_model_name', 'unknown')}`")
    st.caption(
        "Predictions are clipped to the documented relevance range "
        f"({artifact['config'].get('clip_min', 1.0)}–{artifact['config'].get('clip_max', 3.0)})."
    )

query = st.text_input(
    "Search query",
    value="cordless drill",
    placeholder="e.g. 20v cordless drill",
)

st.subheader("Candidate products")
st.write("Add or edit rows. Every candidate with a product title is scored.")

initial_candidates = pd.DataFrame(
    [
        {
            "Product title": "DEWALT 20V MAX Cordless Drill/Driver Kit",
            "Description": "Compact cordless drill with two batteries and charger.",
            "Brand": "DEWALT",
        },
        {
            "Product title": "Makita 18V LXT Hammer Driver-Drill",
            "Description": "Brushless hammer drill for concrete and wood.",
            "Brand": "Makita",
        },
        {
            "Product title": "RIDGID 18V Cordless Oscillating Multi-Tool",
            "Description": "Multi-tool for cutting and sanding.",
            "Brand": "RIDGID",
        },
    ]
)

edited = st.data_editor(
    initial_candidates,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    key="candidate_editor",
)

if st.button("Score and rank candidates", type="primary"):
    candidates = [
        {
            "title": row.get("Product title", ""),
            "description": row.get("Description", ""),
            "brand": row.get("Brand", ""),
        }
        for row in edited.to_dict("records")
    ]

    try:
        with st.spinner("Scoring candidates with the trained model..."):
            embedding_model = get_embedding_model(resolve_embedding_model_name(artifact))
            results = score_candidates(query, candidates, artifact, embedding_model)
    except Exception as exc:  # pragma: no cover - UI error path
        st.error(f"Could not score candidates: {exc}")
    else:
        st.subheader("Ranked results")
        display = results[
            [
                "rank",
                "product_title",
                "product_brand",
                "predicted_relevance",
            ]
        ].rename(
            columns={
                "rank": "Rank",
                "product_title": "Product",
                "product_brand": "Brand",
                "predicted_relevance": "Predicted relevance",
            }
        )
        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Predicted relevance": st.column_config.NumberColumn(
                    format="%.3f", min_value=1.0, max_value=3.0
                )
            },
        )

        best = results.iloc[0]
        st.success(
            f"Top match: **{best['product_title']}** "
            f"(predicted relevance {best['predicted_relevance']:.3f})"
        )

        with st.expander("Show candidate details"):
            st.dataframe(
                results[
                    [
                        "rank",
                        "product_title",
                        "product_description",
                        "product_brand",
                        "predicted_relevance",
                    ]
                ].rename(
                    columns={
                        "rank": "Rank",
                        "product_title": "Product",
                        "product_description": "Description",
                        "product_brand": "Brand",
                        "predicted_relevance": "Predicted relevance",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
