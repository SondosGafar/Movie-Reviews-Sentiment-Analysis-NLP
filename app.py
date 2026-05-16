"""
Streamlit front-end for the saved IMDb sentiment model.
Run: streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from sentiment_inference import predict_one

EXAMPLE_POS_AMAZING = (
    "This movie was amazing and the acting was excellent - a must-see."
)
EXAMPLE_NEG_BORING = (
    "The film was boring, slow, and a complete waste of time. I walked out."
)
EXAMPLE_MIXED = (
    "The story was okay but the ending was disappointing and predictable."
)


def main() -> None:
    st.set_page_config(page_title="IMDb Sentiment Analysis", layout="centered")
    st.title("IMDb Sentiment Analysis System")
    st.write(
        "Type or paste a movie review below. The model outputs **Positive** or **Negative** "
        "with a confidence score when available."
    )

    if "review_text" not in st.session_state:
        st.session_state.review_text = ""

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Example: positive"):
            st.session_state.review_text = EXAMPLE_POS_AMAZING
    with c2:
        if st.button("Example: negative"):
            st.session_state.review_text = EXAMPLE_NEG_BORING
    with c3:
        if st.button("Example: mixed"):
            st.session_state.review_text = EXAMPLE_MIXED

    text = st.text_area(
        "Movie review",
        height=180,
        placeholder="Write your review here...",
        key="review_text",
    )

    if st.button("Predict"):
        if not text or not str(text).strip():
            st.warning("Please enter some review text first.")
        else:
            try:
                r = predict_one(str(text).strip())
            except FileNotFoundError as e:
                st.error(str(e))
                st.info("Run `python save_best_model.py` once to create the model files.")
                return
            st.subheader(f"Prediction: **{r['label']}**")
            if r["prob_positive"] is not None:
                st.metric("P(positive)", f"{r['prob_positive']:.4f}")
                st.caption(
                    f"P(negative) = {r['prob_negative']:.4f}  ·  model: {r['category']}"
                )
            else:
                st.caption(f"Confidence not available · model: {r['category']}")


if __name__ == "__main__":
    main()
