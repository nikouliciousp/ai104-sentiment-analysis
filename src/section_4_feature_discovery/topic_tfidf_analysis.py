"""
TF-IDF analysis per topic.

Computes term importance using TF-IDF in two ways:
  1. Global fit  — one vectoriser on the full corpus; mean TF-IDF per term
                   within each topic (enables cross-topic comparison).
  2. Local fit   — separate vectoriser per topic (highlights terms distinctive
                   within that topic's own document collection).

Input : data/features/hackernews_topic_features_dataset.csv
Output: results/tables/section_4/topic_tfidf_global_full.csv
        results/tables/section_4/topic_tfidf_global_top20.csv
        results/tables/section_4/topic_tfidf_local_top20.csv
        results/tables/section_4/topic_tfidf_frequency_comparison.csv
        results/figures/section_4/topic_tfidf_global_top15.png
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INPUT_PATH = os.path.join(
    BASE_DIR, "data", "features", "hackernews_topic_features_dataset.csv"
)

TABLES_DIR = os.path.join(BASE_DIR, "results", "tables", "section_4")
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures", "section_4")

GLOBAL_FULL_OUTPUT = os.path.join(TABLES_DIR, "topic_tfidf_global_full.csv")
GLOBAL_TOP_OUTPUT = os.path.join(TABLES_DIR, "topic_tfidf_global_top20.csv")
LOCAL_TOP_OUTPUT = os.path.join(TABLES_DIR, "topic_tfidf_local_top20.csv")
COMPARISON_OUTPUT = os.path.join(TABLES_DIR, "topic_tfidf_frequency_comparison.csv")
FIGURE_OUTPUT = os.path.join(FIGURES_DIR, "topic_tfidf_global_top15.png")

TEXT_COLUMN = "text_no_stopwords"

TOP_N_TABLE = 20
TOP_N_FIGURE = 15

# Aligned with sentiment_classification.py for cross-pipeline consistency.
VECTORIZER_PARAMS = {
    "lowercase": True,
    "ngram_range": (1, 1),
    "min_df": 2,
    "max_df": 0.9,
}

TOPIC_COLORS = {
    "Artificial Intelligence": "#4C78A8",
    "Cryptocurrency": "#F58518",
    "Climate Change": "#54A24B",
    "Cybersecurity": "#E45756",
}


# --------------------------------------------------------------------------
# Global TF-IDF
# --------------------------------------------------------------------------

def compute_global_tfidf_stats(df):
    """
    Fit TF-IDF on the full corpus and aggregate mean scores per topic.

    Also computes topic_distinctiveness_ratio:
        mean TF-IDF in topic / mean TF-IDF in all other topics
    """
    texts = df[TEXT_COLUMN].tolist()
    topics = df["topic"].tolist()

    vectorizer = TfidfVectorizer(**VECTORIZER_PARAMS)
    matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    topic_series = pd.Series(topics)
    unique_topics = sorted(topic_series.unique())

    rows = []

    for topic in unique_topics:
        topic_mask = (topic_series == topic).to_numpy()
        other_mask = ~topic_mask

        topic_matrix = matrix[topic_mask]
        other_matrix = matrix[other_mask]

        topic_mean = np.asarray(topic_matrix.mean(axis=0)).ravel()
        other_mean = np.asarray(other_matrix.mean(axis=0)).ravel()

        topic_doc_count = int(topic_mask.sum())
        other_doc_count = int(other_mask.sum())

        for index, term in enumerate(feature_names):
            topic_score = float(topic_mean[index])
            other_score = float(other_mean[index])

            if other_score > 0:
                distinctiveness = topic_score / other_score
            elif topic_score > 0:
                distinctiveness = float("inf")
            else:
                distinctiveness = 0.0

            rows.append({
                "topic": topic,
                "term": term,
                "mean_tfidf_in_topic": round(topic_score, 6),
                "mean_tfidf_in_other": round(other_score, 6),
                "topic_distinctiveness_ratio": round(distinctiveness, 4),
                "topic_doc_count": topic_doc_count,
                "other_doc_count": other_doc_count,
                "fit_scope": "global",
            })

    stats_df = pd.DataFrame(rows)
    stats_df = stats_df.sort_values(
        ["topic", "mean_tfidf_in_topic"],
        ascending=[True, False],
    ).reset_index(drop=True)

    return stats_df, vectorizer, matrix


def compute_local_tfidf_stats(df):
    """
    Fit a separate TF-IDF vectoriser on each topic's documents.

    Highlights terms that are frequent relative to that topic's own vocabulary.
    """
    rows = []

    for topic, topic_df in df.groupby("topic", sort=True):
        texts = topic_df[TEXT_COLUMN].tolist()

        vectorizer = TfidfVectorizer(**VECTORIZER_PARAMS)
        matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        mean_scores = np.asarray(matrix.mean(axis=0)).ravel()

        for index, term in enumerate(feature_names):
            rows.append({
                "topic": topic,
                "term": term,
                "mean_tfidf_local": round(float(mean_scores[index]), 6),
                "topic_doc_count": len(topic_df),
                "fit_scope": "local",
            })

    stats_df = pd.DataFrame(rows)
    stats_df = stats_df.sort_values(
        ["topic", "mean_tfidf_local"],
        ascending=[True, False],
    ).reset_index(drop=True)

    return stats_df


def build_top_terms_table(stats_df, score_column, top_n):
    """Select top N terms per topic by a score column."""
    return (
        stats_df
        .groupby("topic", group_keys=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def build_frequency_comparison(global_stats_df, freq_top_path):
    """
    Compare TF-IDF ranking with term-frequency ranking for the same terms.

    Helps justify why TF-IDF surfaces different terms than raw count.
    """
    if not os.path.exists(freq_top_path):
        return None

    freq_df = pd.read_csv(freq_top_path)

    freq_rank = freq_df.copy()
    freq_rank["frequency_rank"] = (
        freq_rank
        .groupby("topic")["topic_term_count"]
        .rank(ascending=False, method="first")
        .astype(int)
    )

    tfidf_top = build_top_terms_table(global_stats_df, "mean_tfidf_in_topic", TOP_N_TABLE)
    tfidf_rank = tfidf_top.copy()
    tfidf_rank["tfidf_rank"] = (
        tfidf_rank
        .groupby("topic")["mean_tfidf_in_topic"]
        .rank(ascending=False, method="first")
        .astype(int)
    )

    comparison = tfidf_rank.merge(
        freq_rank[["topic", "term", "topic_term_count", "frequency_rank"]],
        on=["topic", "term"],
        how="left",
    )

    comparison["frequency_rank"] = comparison["frequency_rank"].fillna(0).astype(int)
    comparison["rank_difference"] = comparison["frequency_rank"] - comparison["tfidf_rank"]

    comparison = comparison.sort_values(
        ["topic", "tfidf_rank"],
        ascending=[True, True],
    ).reset_index(drop=True)

    return comparison


def plot_global_tfidf(top_df, output_path, top_n):
    """Bar chart of top terms per topic by mean global TF-IDF."""
    topics = sorted(top_df["topic"].unique())
    n_topics = len(topics)

    fig, axes = plt.subplots(
        nrows=n_topics,
        ncols=1,
        figsize=(10, 3.5 * n_topics),
        squeeze=False,
    )

    for ax, topic in zip(axes.flatten(), topics):
        topic_data = (
            top_df[top_df["topic"] == topic]
            .sort_values("mean_tfidf_in_topic", ascending=True)
            .tail(top_n)
        )

        color = TOPIC_COLORS.get(topic, "#333333")

        ax.barh(
            topic_data["term"],
            topic_data["mean_tfidf_in_topic"],
            color=color,
            alpha=0.85,
        )
        ax.set_title(f"{topic} — top {top_n} terms (mean global TF-IDF)")
        ax.set_xlabel("Mean TF-IDF score within topic")
        ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Topic features dataset not found: {INPUT_PATH}\n"
            "Run topic_text_preparation.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = [TEXT_COLUMN, "topic"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=[TEXT_COLUMN, "topic"]).reset_index(drop=True)
    df = df[df[TEXT_COLUMN].str.strip() != ""].reset_index(drop=True)

    print(f"Loaded {len(df)} records")
    print(f"Vectoriser params: {VECTORIZER_PARAMS}")
    print()

    global_stats_df, vectorizer, matrix = compute_global_tfidf_stats(df)
    local_stats_df = compute_local_tfidf_stats(df)

    global_top_df = build_top_terms_table(
        global_stats_df, "mean_tfidf_in_topic", TOP_N_TABLE
    )
    local_top_df = build_top_terms_table(
        local_stats_df, "mean_tfidf_local", TOP_N_TABLE
    )

    global_stats_df.to_csv(GLOBAL_FULL_OUTPUT, index=False)
    global_top_df.to_csv(GLOBAL_TOP_OUTPUT, index=False)
    local_top_df.to_csv(LOCAL_TOP_OUTPUT, index=False)

    freq_top_path = os.path.join(TABLES_DIR, "topic_term_frequency_top20.csv")
    comparison_df = build_frequency_comparison(global_stats_df, freq_top_path)
    if comparison_df is not None:
        comparison_df.to_csv(COMPARISON_OUTPUT, index=False)

    plot_global_tfidf(global_top_df, FIGURE_OUTPUT, TOP_N_FIGURE)

    print(f"Global vocabulary size: {len(vectorizer.get_feature_names_out())}")
    print(f"TF-IDF matrix shape: {matrix.shape}")
    print()
    print("Top 10 terms per topic (mean global TF-IDF):")
    print()
    for topic in sorted(df["topic"].unique()):
        print(f"--- {topic} ---")
        topic_top = global_top_df[global_top_df["topic"] == topic].head(10)
        for _, row in topic_top.iterrows():
            print(
                f"  {row['term']:20s} "
                f"tfidf={row['mean_tfidf_in_topic']:.4f}  "
                f"distinctiveness={row['topic_distinctiveness_ratio']:.2f}"
            )
        print()

    print("Saved:")
    print(GLOBAL_FULL_OUTPUT)
    print(GLOBAL_TOP_OUTPUT)
    print(LOCAL_TOP_OUTPUT)
    if comparison_df is not None:
        print(COMPARISON_OUTPUT)
    print(FIGURE_OUTPUT)


if __name__ == "__main__":
    main()
