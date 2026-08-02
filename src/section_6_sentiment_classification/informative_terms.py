"""
Section 7d - Most Informative Keywords and Bi-grams per Sentiment

Identifies the terms and bi-grams most strongly associated with each sentiment
category, and distinguishes terms that are merely frequent from terms that are
genuinely discriminative.

Method
------
The best-performing sentiment model (Naive Bayes, Section 6c) exposes, for every
term, the log-probability of that term under each class: model.feature_log_prob_.

A term that is merely frequent (for example a topic name that appears across all
sentiments) has similar log-probabilities under every class, so it does not help
discrimination. A term genuinely associated with a sentiment has a much higher
log-probability under that one class than under the others.

We therefore rank terms not by raw frequency but by a discrimination score:

    score(term, class) = log P(term | class) - mean_{other classes} log P(term | class)

A high positive score means the term is characteristic of that class specifically.
This directly answers the assignment's requirement to distinguish frequent terms
from meaningfully associated terms.

The analysis is run twice: once for unigrams and once for bi-grams (ngram=(2,2)).

Input : data/modeling/sentiment_data.csv        (preferred)
        or data/modeling/hackernews_modeling_dataset.csv (fallback)
Output: results/tables/section_7/informative_unigrams_by_sentiment.csv
        results/tables/section_7/informative_bigrams_by_sentiment.csv
        results/tables/section_7/frequent_vs_discriminative.csv
        results/figures/section_7/informative_unigrams_by_sentiment.png
        results/figures/section_7/informative_bigrams_by_sentiment.png
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# HTML artefacts and non-content tokens that survived cleaning. These are
# removed so that the analysis reflects genuine vocabulary, not markup.
NOISE_TOKENS = {
    "href", "rel", "nofollow", "noreferrer", "http", "https", "www",
    "com", "org", "html", "amp", "gt", "lt", "quot",
    "don", "isn", "didn", "doesn", "wasn", "aren", "won", "couldn",
    "wouldn", "shouldn", "hasn", "haven", "wasn",
    "ll", "ve", "re",
}
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PRIMARY_INPUT = os.path.join(BASE_DIR, "data", "modeling", "sentiment_data.csv")
FALLBACK_INPUT = os.path.join(BASE_DIR, "data", "modeling", "hackernews_modeling_dataset.csv")

TABLES_DIR = os.path.join(BASE_DIR, "results", "tables", "section_7")
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures", "section_7")

LABELS = ["negative", "neutral", "positive"]
LABELS_GR = {"negative": "Αρνητικό", "neutral": "Ουδέτερο", "positive": "Θετικό"}
COLORS = {"negative": "#D9534F", "neutral": "#8FA9C4", "positive": "#6BBF59"}

TOP_N = 15


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_dataset():
    if os.path.exists(PRIMARY_INPUT):
        df = pd.read_csv(PRIMARY_INPUT)
        source = PRIMARY_INPUT
    else:
        df = pd.read_csv(FALLBACK_INPUT)
        source = FALLBACK_INPUT

    if "sentiment_3class" not in df.columns:
        df["sentiment_3class"] = df["final_sentiment"]

    df = df.dropna(subset=["text_clean"]).reset_index(drop=True)
    print(f"Loaded {len(df)} records from {os.path.basename(source)}")
    return df


# --------------------------------------------------------------------------
# Core analysis
# --------------------------------------------------------------------------

def analyse(df, ngram_range, label, min_df=3):
    """
    Fit Naive Bayes on the given n-gram representation and extract, per class,
    the most discriminative terms.

    Returns a tidy DataFrame with columns:
        class, term, log_prob, discrimination_score, rank
    """

    # Stopwords are removed here. This is the mechanism by which merely frequent
    # function words (the, is, a) are prevented from dominating. Content words
    # that survive are then ranked by discrimination, not by frequency.
    vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        stop_words=list(ENGLISH_STOP_WORDS | NOISE_TOKENS),
        min_df=min_df,
        lowercase=True,
        token_pattern=r"(?u)\b[a-z][a-z]+\b",
    )

    X = vectorizer.fit_transform(df["text_clean"])
    y = df["sentiment_3class"]
    terms = np.array(vectorizer.get_feature_names_out())

    # Weighted Naive Bayes, consistent with the best model of Section 6c.
    sample_weight = compute_sample_weight(class_weight="balanced", y=y)
    model = MultinomialNB()
    model.fit(X, y, sample_weight=sample_weight)

    # feature_log_prob_ has shape (n_classes, n_terms)
    log_prob = model.feature_log_prob_
    classes = list(model.classes_)

    rows = []
    for class_index, class_name in enumerate(classes):
        this_class = log_prob[class_index]
        other_classes = np.delete(log_prob, class_index, axis=0).mean(axis=0)
        discrimination = this_class - other_classes

        order = np.argsort(discrimination)[::-1][:TOP_N]
        for rank, term_index in enumerate(order, start=1):
            rows.append({
                "class": class_name,
                "term": terms[term_index],
                "log_prob": round(float(this_class[term_index]), 4),
                "discrimination_score": round(float(discrimination[term_index]), 4),
                "rank": rank,
            })

    result = pd.DataFrame(rows)
    print(f"  {label}: vocabulary size {len(terms)}")
    return result, model, terms, log_prob, classes


def frequent_vs_discriminative(df):
    """
    Produce a table contrasting the most FREQUENT terms with the most
    DISCRIMINATIVE terms, to make the distinction explicit and quantitative.
    """

    vectorizer = CountVectorizer(
        ngram_range=(1, 1),
        stop_words=list(ENGLISH_STOP_WORDS | NOISE_TOKENS),
        min_df=3,
        lowercase=True,
        token_pattern=r"(?u)\b[a-z][a-z]+\b",
    )
    X = vectorizer.fit_transform(df["text_clean"])
    terms = np.array(vectorizer.get_feature_names_out())

    total_counts = np.asarray(X.sum(axis=0)).ravel()

    # Most frequent terms overall
    freq_order = np.argsort(total_counts)[::-1][:TOP_N]

    y = df["sentiment_3class"]
    sample_weight = compute_sample_weight(class_weight="balanced", y=y)
    model = MultinomialNB()
    model.fit(X, y, sample_weight=sample_weight)
    log_prob = model.feature_log_prob_

    # For each frequent term, how discriminative is it?
    # Spread = max class log-prob minus min class log-prob. Low spread means the
    # term is distributed evenly across sentiments, i.e. frequent but not
    # discriminative.
    rows = []
    for term_index in freq_order:
        class_log_probs = log_prob[:, term_index]
        spread = float(class_log_probs.max() - class_log_probs.min())
        rows.append({
            "term": terms[term_index],
            "total_count": int(total_counts[term_index]),
            "log_prob_spread": round(spread, 4),
            "interpretation": "διακριτική" if spread > 1.0 else "απλώς συχνή",
        })

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def make_figure(result, title, path):
    """Horizontal bar charts, one panel per sentiment class."""

    figure, axes = plt.subplots(1, 3, figsize=(16, 6))

    for index, class_name in enumerate(LABELS):
        subset = result[result["class"] == class_name].sort_values(
            "discrimination_score", ascending=True
        )

        axes[index].barh(
            subset["term"],
            subset["discrimination_score"],
            color=COLORS[class_name],
        )
        axes[index].set_title(LABELS_GR[class_name])
        axes[index].set_xlabel("Δείκτης διάκρισης (log-prob)")
        axes[index].tick_params(axis="y", labelsize=9)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    df = load_dataset()

    print("\nUnigrams:")
    unigrams, _, _, _, _ = analyse(df, (1, 1), "unigrams")

    print("Bi-grams:")
    bigrams, _, _, _, _ = analyse(df, (2, 2), "bigrams", min_df=2)

    print("Frequent vs discriminative:")
    contrast = frequent_vs_discriminative(df)

    # Save tables
    unigrams.to_csv(
        os.path.join(TABLES_DIR, "informative_unigrams_by_sentiment.csv"),
        index=False, encoding="utf-8-sig",
    )
    bigrams.to_csv(
        os.path.join(TABLES_DIR, "informative_bigrams_by_sentiment.csv"),
        index=False, encoding="utf-8-sig",
    )
    contrast.to_csv(
        os.path.join(TABLES_DIR, "frequent_vs_discriminative.csv"),
        index=False, encoding="utf-8-sig",
    )

    # Figures
    make_figure(
        unigrams,
        "Πιο διακριτικές λέξεις ανά συναίσθημα (unigrams)",
        os.path.join(FIGURES_DIR, "informative_unigrams_by_sentiment.png"),
    )
    make_figure(
        bigrams,
        "Πιο διακριτικά bi-grams ανά συναίσθημα",
        os.path.join(FIGURES_DIR, "informative_bigrams_by_sentiment.png"),
    )

    # Console summary
    print("\n" + "=" * 60)
    print("TOP DISCRIMINATIVE UNIGRAMS PER CLASS")
    print("=" * 60)
    for class_name in LABELS:
        top = unigrams[unigrams["class"] == class_name].head(10)
        terms_list = ", ".join(top["term"].tolist())
        print(f"\n{class_name}:")
        print(f"  {terms_list}")

    print("\n" + "=" * 60)
    print("TOP DISCRIMINATIVE BI-GRAMS PER CLASS")
    print("=" * 60)
    for class_name in LABELS:
        top = bigrams[bigrams["class"] == class_name].head(8)
        terms_list = ", ".join(top["term"].tolist())
        print(f"\n{class_name}:")
        print(f"  {terms_list}")

    print("\n" + "=" * 60)
    print("FREQUENT vs DISCRIMINATIVE (top frequent terms)")
    print("=" * 60)
    print(contrast.to_string(index=False))

    print("\nSaved tables to", TABLES_DIR)
    print("Saved figures to", FIGURES_DIR)


if __name__ == "__main__":
    main()