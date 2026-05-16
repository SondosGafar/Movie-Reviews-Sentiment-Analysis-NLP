"""
Text preprocessing aligned with `sentiment_analysis_update.ipynb`.
Used when saving/loading the exported sentiment model for inference.
"""

from __future__ import annotations

import re
import unicodedata

import nltk
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer

# NLTK resources required for `preprocess_text`
for _pkg in (
    "stopwords",
    "punkt",
    "punkt_tab",
    "wordnet",
    "omw-1.4",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
):
    nltk.download(_pkg, quiet=True)

STOPWORDS = set(stopwords.words("english"))
NEGATIONS_TO_KEEP = {"no", "not", "never", "nor"}
STOPWORDS = STOPWORDS - NEGATIONS_TO_KEEP

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

CONTRACTIONS = {
    "can't": "can not",
    "won't": "will not",
    "n't": " not",
    "i'm": "i am",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "what's": "what is",
    "don't": "do not",
    "didn't": "did not",
    "doesn't": "does not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "wouldn't": "would not",
}


def expand_contractions(text: str) -> str:
    t = text
    t_lower = t.lower()
    for k in sorted(CONTRACTIONS.keys(), key=len, reverse=True):
        t_lower = t_lower.replace(k, CONTRACTIONS[k])
    return t_lower


def strip_html(text: str) -> str:
    return BeautifulSoup(text, "html.parser").get_text(" ")


def remove_accented(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore")


URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
NON_LETTER_RE = re.compile(r"[^a-zA-Z\s]")
MULTISPACE_RE = re.compile(r"\s+")


def get_wordnet_pos(treebank_tag: str):
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("N"):
        return wordnet.NOUN
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def preprocess_text(text: str, variant: str = "none", do_contractions: bool = True) -> str:
    t = str(text)
    t = strip_html(t)
    if do_contractions:
        t = expand_contractions(t)
    else:
        t = t.lower()
    t = URL_RE.sub(" ", t)
    t = remove_accented(t)
    t = NON_LETTER_RE.sub(" ", t)
    t = MULTISPACE_RE.sub(" ", t).strip()

    tokens = nltk.word_tokenize(t)
    tokens = [tok for tok in tokens if tok]
    tokens = [tok for tok in tokens if tok not in STOPWORDS]

    if variant == "stem":
        tokens = [stemmer.stem(tok) for tok in tokens]
    elif variant == "lemma":
        pos_tags = nltk.pos_tag(tokens)
        tokens = [lemmatizer.lemmatize(tok, get_wordnet_pos(tag)) for tok, tag in pos_tags]
    elif variant != "none":
        raise ValueError("variant must be one of: 'none', 'stem', 'lemma'")

    return " ".join(tokens)


def preprocess_series(texts: pd.Series, variant: str) -> pd.Series:
    return texts.map(lambda t: preprocess_text(t, variant=variant, do_contractions=True))
