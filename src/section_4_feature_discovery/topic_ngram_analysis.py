"""
Bigram (n-gram) analysis per topic.

Identifies important two-word phrases using:
  1. Raw bigram frequency counts per topic
  2. Global bigram TF-IDF with per-topic aggregation
  3. Local bigram TF-IDF fitted within each topic

Short HN comments often express topic meaning through phrases such as
"machine learning" or "climate change", so bigrams are analysed separately
from unigram term frequency and TF-IDF.

Input : data/features/hackernews_topic_features_dataset.csv
Output: results/tables/section_4/topic_bigram_frequency_top20.csv
        results/tables/section_4/topic_bigram_frequency_full.csv
        results/tables/section_4/topic_bigram_tfidf_global_top20.csv
        results/tables/section_4/topic_bigram_tfidf_global_full.csv
        results/tables/section_4/topic_bigram_tfidf_local_top20.csv
        results/tables/section_4/topic_bigram_coverage_summary.csv
        results/figures/section_4/topic_bigram_frequency_top15.png
        results/figures/section_4/topic_bigram_tfidf_global_top15.png
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INPUT_PATH = os.path.join(
    BASE_DIR, "data", "features", "hackernews_topic_features_dataset.csv"
)

TABLES_DIR = os.path.join(BASE_DIR, "results", "tables", "section_4")
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures", "section_4")

FREQ_FULL_OUTPUT = os.path.join(TABLES_DIR, "topic_bigram_frequency_full.csv")
FREQ_TOP_OUTPUT = os.path.join(TABLES_DIR, "topic_bigram_frequency_top20.csv")
GLOBAL_FULL_OUTPUT = os.path.join(TABLES_DIR, "topic_bigram_tfidf_global_full.csv")
GLOBAL_TOP_OUTPUT = os.path.join(TABLES_DIR, "topic_bigram_tfidf_global_top20.csv")
LOCAL_TOP_OUTPUT = os.path.join(TABLES_DIR, "topic_bigram_tfidf_local_top20.csv")
COVERAGE_OUTPUT = os.path.join(TABLES_DIR, "topic_bigram_coverage_summary.csv")
FREQ_FIGURE_OUTPUT = os.path.join(FIGURES_DIR, "topic_bigram_frequency_top15.png")
TFIDF_FIGURE_OUTPUT = os.path.join(FIGURES_DIR, "topic_bigram_tfidf_global_top15.png")

TEXT_COLUMN = "text_no_stopwords"

TOP_N_TABLE = 20
TOP_N_FIGURE = 15

BIGRAM_VECTORIZER_PARAMS = {
    "lowercase": True,
    "ngram_range": (2, 2),
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
# Helpers
# --------------------------------------------------------------------------

def build_top_n_table(stats_df, score_column, top_n):
    return (
        stats_df
        .groupby("topic", group_keys=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def compute_bigram_coverage(df, vectorizer, matrix):
    """Summarise how many documents contain at least one retained bigram."""
    topics = df["topic"].tolist()
    topic_series = pd.Series(topics)
    rows = []

    for topic in sorted(topic_series.unique()):
        topic_mask = (topic_series == topic).to_numpy()
        topic_matrix = matrix[topic_mask]

        docs_with_bigram = int(np.count_nonzero(topic_matrix.sum(axis=1)))
        topic_doc_count = int(topic_mask.sum())
        coverage_pct = round(100 * docs_with_bigram / topic_doc_count, 2)

        rows.append({
            "topic": topic,
            "topic_doc_count": topic_doc_count,
            "docs_with_bigram": docs_with_bigram,
            "bigram_doc_coverage_pct": coverage_pct,
            "retained_bigram_count": len(vectorizer.get_feature_names_out()),
        })

    return pd.DataFrame(rows)


def compute_bigram_frequency_stats(df):
    """
    Fit a global CountVectorizer on bigrams and aggregate counts per topic.

    Also reports document frequency: number of topic documents containing
    each bigram at least once.
    """
    texts = df[TEXT_COLUMN].tolist()
    topics = df["topic"].tolist()

    vectorizer = CountVectorizer(**BIGRAM_VECTORIZER_PARAMS)
    matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    topic_series = pd.Series(topics)
    rows = []

    for topic in sorted(topic_series.unique()):
        topic_mask = (topic_series == topic).to_numpy()
        topic_matrix = matrix[topic_mask]
        topic_doc_count = int(topic_mask.sum())

        total_counts = np.asarray(topic_matrix.sum(axis=0)).ravel()
        doc_freq = np.asarray((topic_matrix > 0).sum(axis=0)).ravel()
        topic_total = int(total_counts.sum())

        for index, bigram in enumerate(feature_names):
            bigram_count = int(total_counts[index])
            bigram_doc_freq = int(doc_freq[index])

            if topic_total > 0:
                bigram_freq = bigram_count / topic_total
            else:
                bigram_freq = 0.0

            if topic_doc_count > 0:
                bigram_doc_ratio = bigram_doc_freq / topic_doc_count
            else:
                bigram_doc_ratio = 0.0

            rows.append({
                "topic": topic,
                "bigram": bigram,
                "bigram_count": bigram_count,
                "bigram_freq": round(bigram_freq, 6),
                "bigram_doc_freq": bigram_doc_freq,
                "bigram_doc_ratio": round(bigram_doc_ratio, 6),
                "topic_doc_count": topic_doc_count,
            })

    stats_df = pd.DataFrame(rows)
    stats_df = stats_df.sort_values(
        ["topic", "bigram_count"],
        ascending=[True, False],
    ).reset_index(drop=True)

    return stats_df, vectorizer, matrix


def compute_global_bigram_tfidf_stats(df):
    """Global bigram TF-IDF with per-topic mean scores and distinctiveness."""
    texts = df[TEXT_COLUMN].tolist()
    topics = df["topic"].tolist()

    vectorizer = TfidfVectorizer(**BIGRAM_VECTORIZER_PARAMS)
    matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    topic_series = pd.Series(topics)
    rows = []

    for topic in sorted(topic_series.unique()):
        topic_mask = (topic_series == topic).to_numpy()
        other_mask = ~topic_mask

        topic_matrix = matrix[topic_mask]
        other_matrix = matrix[other_mask]

        topic_mean = np.asarray(topic_matrix.mean(axis=0)).ravel()
        other_mean = np.asarray(other_matrix.mean(axis=0)).ravel()

        topic_doc_count = int(topic_mask.sum())
        other_doc_count = int(other_mask.sum())

        for index, bigram in enumerate(feature_names):
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
                "bigram": bigram,
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


def compute_local_bigram_tfidf_stats(df):
    """Separate bigram TF-IDF vectoriser per topic."""
    rows = []

    for topic, topic_df in df.groupby("topic", sort=True):
        texts = topic_df[TEXT_COLUMN].tolist()

        vectorizer = TfidfVectorizer(**BIGRAM_VECTORIZER_PARAMS)
        matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        mean_scores = np.asarray(matrix.mean(axis=0)).ravel()

        for index, bigram in enumerate(feature_names):
            rows.append({
                "topic": topic,
                "bigram": bigram,
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


def plot_top_bigrams(top_df, score_column, output_path, top_n, title_suffix):
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
            .sort_values(score_column, ascending=True)
            .tail(top_n)
        )

        color = TOPIC_COLORS.get(topic, "#333333")

        ax.barh(
            topic_data["bigram"],
            topic_data[score_column],
            color=color,
            alpha=0.85,
        )
        ax.set_title(f"{topic} — top {top_n} bigrams ({title_suffix})")
        ax.set_xlabel(score_column.replace("_", " "))
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
    print(f"Bigram vectoriser params: {BIGRAM_VECTORIZER_PARAMS}")
    print()

    freq_stats_df, count_vectorizer, count_matrix = compute_bigram_frequency_stats(df)
    global_stats_df, tfidf_vectorizer, tfidf_matrix = compute_global_bigram_tfidf_stats(df)
    local_stats_df = compute_local_bigram_tfidf_stats(df)
    coverage_df = compute_bigram_coverage(df, count_vectorizer, count_matrix)

    freq_top_df = build_top_n_table(freq_stats_df, "bigram_count", TOP_N_TABLE)
    global_top_df = build_top_n_table(global_stats_df, "mean_tfidf_in_topic", TOP_N_TABLE)
    local_top_df = build_top_n_table(local_stats_df, "mean_tfidf_local", TOP_N_TABLE)

    freq_stats_df.to_csv(FREQ_FULL_OUTPUT, index=False)
    freq_top_df.to_csv(FREQ_TOP_OUTPUT, index=False)
    global_stats_df.to_csv(GLOBAL_FULL_OUTPUT, index=False)
    global_top_df.to_csv(GLOBAL_TOP_OUTPUT, index=False)
    local_top_df.to_csv(LOCAL_TOP_OUTPUT, index=False)
    coverage_df.to_csv(COVERAGE_OUTPUT, index=False)

    plot_top_bigrams(
        freq_top_df,
        "bigram_count",
        FREQ_FIGURE_OUTPUT,
        TOP_N_FIGURE,
        "raw count",
    )
    plot_top_bigrams(
        global_top_df,
        "mean_tfidf_in_topic",
        TFIDF_FIGURE_OUTPUT,
        TOP_N_FIGURE,
        "mean global TF-IDF",
    )

    print("Bigram coverage per topic:")
    print(coverage_df.to_string(index=False))
    print()
    print(f"Retained bigrams (global): {len(count_vectorizer.get_feature_names_out())}")
    print(f"Count matrix shape: {count_matrix.shape}")
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print()
    print("Top 10 bigrams per topic (mean global TF-IDF):")
    print()
    for topic in sorted(df["topic"].unique()):
        print(f"--- {topic} ---")
        topic_top = global_top_df[global_top_df["topic"] == topic].head(10)
        for _, row in topic_top.iterrows():
            print(
                f"  {row['bigram']:28s} "
                f"tfidf={row['mean_tfidf_in_topic']:.4f}  "
                f"distinctiveness={row['topic_distinctiveness_ratio']:.2f}"
            )
        print()

    print("Saved:")
    print(FREQ_FULL_OUTPUT)
    print(FREQ_TOP_OUTPUT)
    print(GLOBAL_FULL_OUTPUT)
    print(GLOBAL_TOP_OUTPUT)
    print(LOCAL_TOP_OUTPUT)
    print(COVERAGE_OUTPUT)
    print(FREQ_FIGURE_OUTPUT)
    print(TFIDF_FIGURE_OUTPUT)


if __name__ == "__main__":
    main()
