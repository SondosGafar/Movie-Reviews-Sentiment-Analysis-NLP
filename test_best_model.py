"""
Interactive test harness for the saved IMDb sentiment model.

Requires `python save_best_model.py` once to create files under `models/`.
"""

from __future__ import annotations

import sys

import numpy as np

from sentiment_inference import predict_batch, predict_one

# Sample reviews (edit freely)
SAMPLE_REVIEWS = [
    "This movie was amazing and the acting was excellent.",
    "The film was boring, slow, and a waste of time.",
    "A masterpiece - I would watch it again tomorrow.",
]


def print_prediction(text: str) -> None:
    r = predict_one(text.strip())
    print("Review:", text)
    print("  ->", r["label"])
    if r["prob_positive"] is not None:
        print(
            f"  P(positive) = {r['prob_positive']:.4f}  |  "
            f"P(negative) = {r['prob_negative']:.4f}"
        )
    print(f"  (model type: {r['category']})")
    print()


def run_samples() -> None:
    print("=== Sample reviews ===\n")
    yhat, ppos, cat = predict_batch(SAMPLE_REVIEWS)
    for txt, yi, p in zip(SAMPLE_REVIEWS, yhat, ppos):
        label = "Positive" if int(yi) == 1 else "Negative"
        print("Review:", txt)
        print("  ->", label)
        if not np.isnan(p):
            print(f"  P(positive) = {float(p):.4f}")
        print(f"  (model type: {cat})")
        print()


def interactive_loop() -> None:
    print("Enter movie review text (empty line to quit).\n")
    while True:
        try:
            line = input("Review> ").strip()
        except EOFError:
            print()
            break
        if not line:
            break
        print_prediction(line)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--samples":
        run_samples()
        return
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_loop()
        return
    run_samples()
    print(
        "Tip: use `python test_best_model.py --interactive` to type your own reviews.\n"
    )


if __name__ == "__main__":
    main()
