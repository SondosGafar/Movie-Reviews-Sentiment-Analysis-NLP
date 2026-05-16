# Running the saved model and GUI

These steps use the **best traditional ML** pipeline from `sentiment_analysis_update.ipynb`
(`prep=none`, TF‑IDF unigrams+bigrams+trigrams, `LinearSVC`).

## 1) Create the saved model files (once)

From the project folder:

```bash
python save_best_model.py
```

This writes `models/best_sentiment_model.joblib` and `models/model_metadata.pkl`.

## 2) Test in the terminal

```bash
python test_best_model.py
```

Use `--no-input` to only print the built-in sample reviews and exit:

```bash
python test_best_model.py --no-input
```

## 3) Streamlit app

Install dependencies (for example):

```bash
pip install -r requirements_app.txt
```

Then:

```bash
streamlit run app.py
```

## Deep learning exports (optional)

If your overall winner is a **Keras** model, save into `models/`:

- `best_dl_model.keras`
- `tokenizer.pkl`
- `model_config.pkl` (dict with at least `dl_prep`, `max_len`, keys matching the notebook)

and set `category` to `Deep Learning` in `model_metadata.pkl`.
