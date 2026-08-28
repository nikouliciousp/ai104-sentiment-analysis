"""
Leakage-safe custom scoring features for topic classification, per topic.

Companion to topic_classification_features.py: that script scores every post
against every topic's lookup and keeps only the best-matching topic's scores
(document_custom_score etc.), which discards how a post scored against the
other topics. This script keeps ALL of them, emitting one column group per
topic (e.g. artificial_intelligence_custom_score, cryptocurrency_custom_score,
...) so a classifier can use the full per-topic signal instead of a single
argmax'd score.

Fitting is identical to topic_classification_features.py: every scoring
component (term frequency, global TF-IDF, bigram TF-IDF, custom_term_score /
custom_bigram_score) is fit on the TRAIN split only, using the same
train/test split (test_size=0.2, random_state=42), so rows line up 1:1 with
topic_classification_features.csv.

Input : data/features/hackernews_topic_features_dataset.csv
Output: data/features/topic_classification_features_per_topic.csv
"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split

from topic_term_frequency import compute_topic_term_stats
from topic_tfidf_analysis import compute_global_tfidf_stats
from topic_ngram_analysis import compute_global_bigram_tfidf_stats
from topic_custom_scoring import (
    build_bigram_score_table,
    build_term_score_table,
    build_term_lookup,
    build_bigram_lookup,
    score_documents_per_topic,
    topic_column_prefix,
)

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INPUT_PATH = os.path.join(
    BASE_DIR, "data", "features", "hackernews_topic_features_dataset.csv"
)

FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")
FEATURES_OUTPUT = os.path.join(FEATURES_DIR, "topic_classification_features_per_topic.csv")

TEXT_COLUMN = "text_no_stopwords"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def main():
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

    # ----------------------------------------------------------------
    # Fit every scoring component on TRAIN ONLY (identical to
    # topic_classification_features.py).
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
    # Score every post against EVERY topic's train-fitted lookup, keeping
    # all topics' scores (not just the best match).
    # ----------------------------------------------------------------
    enriched_df = score_documents_per_topic(df, term_lookup, bigram_lookup)
    enriched_df["split"] = df["split"].values

    topics = sorted(term_lookup.keys())
    per_topic_columns = [
        f"{topic_column_prefix(topic)}_{suffix}"
        for topic in topics
        for suffix in [
            "unigram_score", "bigram_score", "positional_score", "custom_score",
            "matched_unigram_count", "matched_bigram_count",
        ]
    ]

    output_columns = [
        "post_id", "item_id", "topic", "final_sentiment", "split",
    ] + per_topic_columns
    output_df = enriched_df[output_columns].copy()

    output_df.to_csv(FEATURES_OUTPUT, index=False)

    print(f"Topics (column prefixes): {[topic_column_prefix(t) for t in topics]}")
    print(f"Output columns: {len(output_df.columns)}")
    print()

    print("Saved:")
    print(FEATURES_OUTPUT)


if __name__ == "__main__":
    main()
