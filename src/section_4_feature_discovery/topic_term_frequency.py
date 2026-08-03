"""
Term frequency analysis per topic.

Computes corpus-wide and topic-specific term frequencies on text with
stopwords removed. Used as input to the custom feature-scoring mechanism.

Input : data/features/hackernews_topic_features_dataset.csv
Output: results/tables/section_4/topic_term_frequency_top20.csv
        results/tables/section_4/topic_term_frequency_full.csv
        results/tables/section_4/corpus_term_frequency.csv
        results/figures/section_4/topic_term_frequency_top15.png
"""

import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from topic_text_preparation import tokenize_for_topic_analysis

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INPUT_PATH = os.path.join(
    BASE_DIR, "data", "features", "hackernews_topic_features_dataset.csv"
)

TABLES_DIR = os.path.join(BASE_DIR, "results", "tables", "section_4")
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures", "section_4")

TOP_TERMS_OUTPUT = os.path.join(TABLES_DIR, "topic_term_frequency_top20.csv")
FULL_TERMS_OUTPUT = os.path.join(TABLES_DIR, "topic_term_frequency_full.csv")
CORPUS_OUTPUT = os.path.join(TABLES_DIR, "corpus_term_frequency.csv")
FIGURE_OUTPUT = os.path.join(FIGURES_DIR, "topic_term_frequency_top15.png")

TEXT_COLUMN = "text_no_stopwords"

TOP_N_TABLE = 20
TOP_N_FIGURE = 15

TOPIC_COLORS = {
    "Artificial Intelligence": "#4C78A8",
    "Cryptocurrency": "#F58518",
    "Climate Change": "#54A24B",
    "Cybersecurity": "#E45756",
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def count_terms(texts):
    """Return a Series of term counts for a list of documents."""
    counts = {}
    for text in texts:
        for token in tokenize_for_topic_analysis(text):
            counts[token] = counts.get(token, 0) + 1
    return pd.Series(counts, dtype="int64").sort_values(ascending=False)


def compute_topic_term_stats(df):
    """
    Compute frequency statistics for every term in every topic.

    Columns produced:
        topic_term_count      : raw count of the term within the topic
        topic_doc_count       : number of documents in the topic
        topic_term_freq       : term count / total term tokens in topic
        corpus_term_freq      : term count / total term tokens in full corpus
        topic_specific_ratio  : topic_term_freq / corpus_term_freq (lift)
    """
    corpus_counts = count_terms(df[TEXT_COLUMN])
    corpus_total = int(corpus_counts.sum())

    corpus_freq = corpus_counts / corpus_total

    rows = []

    for topic, topic_df in df.groupby("topic", sort=True):
        topic_counts = count_terms(topic_df[TEXT_COLUMN])
        topic_total = int(topic_counts.sum())
        topic_doc_count = len(topic_df)

        topic_freq = topic_counts / topic_total

        for term, term_count in topic_counts.items():
            corpus_count = int(corpus_counts.get(term, 0))
            corpus_term_freq = float(corpus_freq.get(term, 0.0))
            topic_term_freq = float(topic_freq[term])

            if corpus_term_freq > 0:
                topic_specific_ratio = topic_term_freq / corpus_term_freq
            else:
                topic_specific_ratio = 0.0

            rows.append({
                "topic": topic,
                "term": term,
                "topic_term_count": int(term_count),
                "corpus_term_count": corpus_count,
                "topic_doc_count": topic_doc_count,
                "topic_term_freq": round(topic_term_freq, 6),
                "corpus_term_freq": round(corpus_term_freq, 6),
                "topic_specific_ratio": round(topic_specific_ratio, 4),
            })

    stats_df = pd.DataFrame(rows)
    stats_df = stats_df.sort_values(
        ["topic", "topic_term_count"],
        ascending=[True, False],
    ).reset_index(drop=True)

    return stats_df, corpus_counts, corpus_total


def build_top_terms_table(stats_df, top_n):
    """Select the top N terms per topic by raw topic_term_count."""
    top_rows = (
        stats_df
        .groupby("topic", group_keys=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return top_rows


def plot_top_terms(top_df, output_path, top_n):
    """Horizontal bar chart of the most frequent terms per topic."""
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
            .sort_values("topic_term_count", ascending=True)
            .tail(top_n)
        )

        color = TOPIC_COLORS.get(topic, "#333333")

        ax.barh(
            topic_data["term"],
            topic_data["topic_term_count"],
            color=color,
            alpha=0.85,
        )
        ax.set_title(f"{topic} — top {top_n} terms (raw count, no stopwords)")
        ax.set_xlabel("Term count within topic")
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
    print(f"Text column: {TEXT_COLUMN}")
    print()
    print("Topic distribution:")
    print(df["topic"].value_counts().to_string())
    print()

    stats_df, corpus_counts, corpus_total = compute_topic_term_stats(df)

    corpus_df = pd.DataFrame({
        "term": corpus_counts.index,
        "corpus_term_count": corpus_counts.values,
        "corpus_term_freq": corpus_counts.values / corpus_total,
    })
    corpus_df["corpus_term_freq"] = corpus_df["corpus_term_freq"].round(6)
    corpus_df = corpus_df.reset_index(drop=True)

    top_df = build_top_terms_table(stats_df, TOP_N_TABLE)

    stats_df.to_csv(FULL_TERMS_OUTPUT, index=False)
    top_df.to_csv(TOP_TERMS_OUTPUT, index=False)
    corpus_df.to_csv(CORPUS_OUTPUT, index=False)

    plot_top_terms(top_df, FIGURE_OUTPUT, TOP_N_FIGURE)

    print("Top 10 terms per topic (by raw count):")
    print()
    for topic in sorted(df["topic"].unique()):
        print(f"--- {topic} ---")
        topic_top = top_df[top_df["topic"] == topic].head(10)
        for _, row in topic_top.iterrows():
            print(
                f"  {row['term']:20s} "
                f"count={row['topic_term_count']:4d}  "
                f"freq={row['topic_term_freq']:.4f}  "
                f"lift={row['topic_specific_ratio']:.2f}"
            )
        print()

    print("Saved:")
    print(FULL_TERMS_OUTPUT)
    print(TOP_TERMS_OUTPUT)
    print(CORPUS_OUTPUT)
    print(FIGURE_OUTPUT)


if __name__ == "__main__":
    main()
