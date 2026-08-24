"""
Leakage-safe custom scoring features for topic classification.

Unlike topic_custom_scoring.py (used for descriptive analysis in Sections
4.1-4.2, fitted on the full corpus), this script fits every scoring
component - term frequency, global TF-IDF, bigram TF-IDF, and the resulting
custom_term_score / custom_bigram_score - using ONLY the training split.
The resulting term/bigram lookup is then applied to both the training and
test posts to produce document-level features, so that test-set statistics
never influence the scores (Section 4.3 requirement).

Reuses the scoring logic and weights from topic_term_frequency.py,
topic_tfidf_analysis.py, topic_ngram_analysis.py and topic_custom_scoring.py
so that the fitted mechanism is identical to Section 4.2, only the fitting
data changes.

Input : data/features/hackernews_topic_features_dataset.csv
Output: data/features/topic_classification_features.csv
        results/tables/section_4/topic_classification_split_summary.csv
        results/tables/section_4/topic_classification_features_summary.csv
"""

import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from topic_term_frequency import compute_topic_term_stats
from topic_tfidf_analysis import compute_global_tfidf_stats
from topic_ngram_analysis import compute_global_bigram_tfidf_stats
from topic_custom_scoring import (
    build_bigram_score_table,
    build_term_score_table,
    build_term_lookup,
    build_bigram_lookup,
    score_documents,
)

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INPUT_PATH = os.path.join(
    BASE_DIR, "data", "features", "hackernews_topic_features_dataset.csv"
)

TABLES_DIR = os.path.join(BASE_DIR, "results", "tables", "section_4")
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures", "section_4")
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")

FEATURES_OUTPUT = os.path.join(FEATURES_DIR, "topic_classification_features.csv")
SPLIT_SUMMARY_OUTPUT = os.path.join(TABLES_DIR, "topic_classification_split_summary.csv")
FEATURES_SUMMARY_OUTPUT = os.path.join(TABLES_DIR, "topic_classification_features_summary.csv")
COVERAGE_FIGURE_OUTPUT = os.path.join(FIGURES_DIR, "topic_classification_split_coverage.png")

TEXT_COLUMN = "text_no_stopwords"
TEST_SIZE = 0.2
RANDOM_STATE = 42

SPLIT_COLORS = {"train": "#4C78A8", "test": "#F58518"}


def plot_split_coverage(features_summary, output_path):
    """Compare train vs test term coverage and score quality side by side."""
    order = ["train", "test"]
    summary = features_summary.set_index("split").loc[order]
    bar_colors = [SPLIT_COLORS[s] for s in order]

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(13, 4))

    ax = axes[0]
    x = range(len(order))
    width = 0.35
    ax.bar([i - width / 2 for i in x], summary["mean_matched_unigram_count"], width,
           label="mean matched unigrams", color="#4C78A8", alpha=0.85)
    ax.bar([i + width / 2 for i in x], summary["mean_matched_bigram_count"], width,
           label="mean matched bigrams", color="#F58518", alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(order)
    ax.set_title("Mean matched terms per post")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(order, summary["mean_document_custom_score"], color=bar_colors, alpha=0.85)
    ax.set_title("Mean document_custom_score")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    ax.bar(order, summary["pct_zero_unigram_match"], color=bar_colors, alpha=0.85)
    ax.set_title("% posts with 0 matched unigrams")
    ax.set_ylabel("%")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Train vs. test coverage under leakage-safe (train-only) fitting", y=1.03)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(FEATURES_DIR, exist_ok=True)

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Topic features dataset not found: {INPUT_PATH}\n"
            "Run topic_text_preparation.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = [TEXT_COLUMN, "topic", "final_sentiment", "post_id"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=[TEXT_COLUMN, "topic"]).reset_index(drop=True)
    df = df[df[TEXT_COLUMN].str.strip() != ""].reset_index(drop=True)

    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["topic"], random_state=RANDOM_STATE
    )

    df["split"] = "test"
    df.loc[train_df.index, "split"] = "train"
    train_df = df[df["split"] == "train"].reset_index(drop=True)

    print(f"Loaded {len(df)} records")
    print(
        f"Train: {len(train_df)}  Test: {len(df) - len(train_df)}  "
        f"(test_size={TEST_SIZE}, random_state={RANDOM_STATE})"
    )
    print()

    split_summary = (
        df.groupby(["topic", "split"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    print("Train/test split per topic:")
    print(split_summary.to_string(index=False))
    print()

    # ----------------------------------------------------------------
    # Fit every scoring component on TRAIN ONLY
    # ----------------------------------------------------------------
    freq_stats_df, _, _ = compute_topic_term_stats(train_df)
    tfidf_stats_df, _, _ = compute_global_tfidf_stats(train_df)
    bigram_tfidf_stats_df, _, _ = compute_global_bigram_tfidf_stats(train_df)

    bigram_scores_df = build_bigram_score_table(bigram_tfidf_stats_df)
    term_scores_df = build_term_score_table(
        train_df, freq_stats_df, tfidf_stats_df, bigram_scores_df
    )

    term_lookup = build_term_lookup(term_scores_df)
    bigram_lookup = build_bigram_lookup(bigram_scores_df)

    # ----------------------------------------------------------------
    # Apply the train-fitted lookup to ALL posts (train + test).
    # Tokens/bigrams unseen in training simply have no lookup entry and
    # are skipped by score_documents, i.e. treated as out-of-vocabulary.
    #
    # use_true_topic=False: score each post against every topic's lookup
    # and keep the best match, rather than indexing by the post's own
    # ground-truth topic. Selecting by the true label would leak the
    # classification target into the feature itself.
    # ----------------------------------------------------------------
    enriched_df = score_documents(df, term_lookup, bigram_lookup, use_true_topic=False)
    enriched_df["split"] = df["split"].values

    output_columns = [
        "post_id", "item_id", "topic", "final_sentiment", "split",
        "document_unigram_score", "document_bigram_score",
        "document_positional_score", "document_custom_score",
        "matched_unigram_count", "matched_bigram_count",
    ]
    output_df = enriched_df[output_columns].copy()

    output_df.to_csv(FEATURES_OUTPUT, index=False)
    split_summary.to_csv(SPLIT_SUMMARY_OUTPUT, index=False)

    output_df["zero_unigram_match"] = output_df["matched_unigram_count"] == 0
    features_summary = (
        output_df
        .groupby("split", as_index=False)
        .agg(
            document_count=("post_id", "count"),
            mean_document_custom_score=("document_custom_score", "mean"),
            mean_matched_unigram_count=("matched_unigram_count", "mean"),
            mean_matched_bigram_count=("matched_bigram_count", "mean"),
            pct_zero_unigram_match=("zero_unigram_match", "mean"),
        )
    )
    features_summary["pct_zero_unigram_match"] = (
        features_summary["pct_zero_unigram_match"] * 100
    ).round(2)
    for col in ["mean_document_custom_score", "mean_matched_unigram_count", "mean_matched_bigram_count"]:
        features_summary[col] = features_summary[col].round(4)

    features_summary.to_csv(FEATURES_SUMMARY_OUTPUT, index=False)

    plot_split_coverage(features_summary, COVERAGE_FIGURE_OUTPUT)

    print("Feature summary by split (train-fitted lookup applied to both):")
    print(features_summary.to_string(index=False))
    print()

    print("Saved:")
    print(FEATURES_OUTPUT)
    print(SPLIT_SUMMARY_OUTPUT)
    print(FEATURES_SUMMARY_OUTPUT)
    print(COVERAGE_FIGURE_OUTPUT)


if __name__ == "__main__":
    main()
