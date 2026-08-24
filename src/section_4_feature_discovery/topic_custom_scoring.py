"""
Custom feature scoring for topic discovery.

Combines multiple scoring signals into term-level and document-level scores:
  - frequency-based scoring
  - TF-IDF values
  - topic-specific frequency (lift)
  - sentiment association
  - positional information
  - bigram scoring
  - document-level scoring

Component scores are min-max normalised within each topic before weighting.
Weights are exported for transparency and report documentation.

score_documents() supports two modes: use_true_topic=True (default here,
for descriptive Section 4.1-4.2 reporting) scores each post against its
own ground-truth topic's lookup tables. use_true_topic=False scores a
post against every topic's lookup tables and keeps the best match - this
mode is required whenever the output feeds a topic classifier, since
indexing by the true label would leak the target into the feature. See
topic_classification_features.py, which uses use_true_topic=False.

Input : data/features/hackernews_topic_features_dataset.csv
        results/tables/section_4/topic_term_frequency_full.csv
        results/tables/section_4/topic_tfidf_global_full.csv
        results/tables/section_4/topic_bigram_tfidf_global_full.csv
Output: data/features/features_enriched.csv
        results/tables/section_4/topic_term_custom_scores.csv
        results/tables/section_4/topic_bigram_custom_scores.csv
        results/tables/section_4/topic_custom_scoring_weights.csv
        results/tables/section_4/document_custom_score_summary.csv
        results/figures/section_4/topic_custom_score_top15.png
"""

import os
import re

import numpy as np
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

FREQ_TABLE_PATH = os.path.join(
    BASE_DIR, "results", "tables", "section_4", "topic_term_frequency_full.csv"
)
TFIDF_TABLE_PATH = os.path.join(
    BASE_DIR, "results", "tables", "section_4", "topic_tfidf_global_full.csv"
)
BIGRAM_TFIDF_TABLE_PATH = os.path.join(
    BASE_DIR, "results", "tables", "section_4", "topic_bigram_tfidf_global_full.csv"
)

TABLES_DIR = os.path.join(BASE_DIR, "results", "tables", "section_4")
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures", "section_4")
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")

TERM_SCORES_OUTPUT = os.path.join(TABLES_DIR, "topic_term_custom_scores.csv")
BIGRAM_SCORES_OUTPUT = os.path.join(TABLES_DIR, "topic_bigram_custom_scores.csv")
WEIGHTS_OUTPUT = os.path.join(TABLES_DIR, "topic_custom_scoring_weights.csv")
DOC_SUMMARY_OUTPUT = os.path.join(TABLES_DIR, "document_custom_score_summary.csv")
ENRICHED_OUTPUT = os.path.join(FEATURES_DIR, "features_enriched.csv")
FIGURE_OUTPUT = os.path.join(FIGURES_DIR, "topic_custom_score_top15.png")

TEXT_COLUMN = "text_no_stopwords"
SENTIMENT_COLUMN = "final_sentiment"
VALID_SENTIMENTS = ["negative", "neutral", "positive"]

TOP_N_FIGURE = 15

# Minimum topic_term_count for a term to receive a custom score. Aligns with
# the min_df=2 already used by TfidfVectorizer elsewhere in this pipeline;
# without it, hapax legomena get topic_specific/sentiment/positional
# components maxed out by small-sample effects despite zero frequency support.
MIN_TERM_COUNT = 2

# Term-level component weights (sum = 1.0)
TERM_COMPONENT_WEIGHTS = {
    "freq_score_norm": 0.20,
    "tfidf_score_norm": 0.25,
    "topic_specific_score_norm": 0.20,
    "sentiment_association_score_norm": 0.15,
    "positional_score_norm": 0.10,
    "bigram_boost_score_norm": 0.10,
}

# Document-level component weights (sum = 1.0)
DOCUMENT_COMPONENT_WEIGHTS = {
    "document_unigram_score": 0.55,
    "document_bigram_score": 0.30,
    "document_positional_score": 0.15,
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

def min_max_normalize(series):
    """Min-max scale to [0, 1]. Constant columns become 0."""
    min_value = float(series.min())
    max_value = float(series.max())

    if max_value == min_value:
        return pd.Series(0.0, index=series.index)

    return (series - min_value) / (max_value - min_value)


def position_weight(position_index):
    """Higher weight for tokens appearing earlier in the document."""
    return 1.0 / (position_index + 1)


def extract_bigrams(tokens):
    """Return consecutive token pairs from a token list."""
    if len(tokens) < 2:
        return []

    return [f"{tokens[index]} {tokens[index + 1]}" for index in range(len(tokens) - 1)]


def term_in_text(term, text):
    """Match a unigram as a whole token."""
    return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", str(text)) is not None


def compute_positional_scores(topic_df):
    """
    Average early-position emphasis for each term within a topic.

    Tokens at the start of a comment receive higher positional weight.
    """
    positional_totals = {}
    positional_counts = {}

    for text in topic_df[TEXT_COLUMN]:
        tokens = tokenize_for_topic_analysis(text)

        for index, token in enumerate(tokens):
            weight = position_weight(index)
            positional_totals[token] = positional_totals.get(token, 0.0) + weight
            positional_counts[token] = positional_counts.get(token, 0) + 1

    scores = {}
    for token, total in positional_totals.items():
        scores[token] = total / positional_counts[token]

    return scores


def compute_sentiment_association_scores(topic_df):
    """
    Measure how strongly a term's sentiment profile diverges from the topic baseline.

    Uses the maximum absolute deviation between term-level and topic-level
    sentiment proportions.
    """
    baseline = topic_df[SENTIMENT_COLUMN].value_counts(normalize=True)
    baseline = baseline.reindex(VALID_SENTIMENTS, fill_value=0.0)

    scores = {}

    for _, row in topic_df.iterrows():
        tokens = set(tokenize_for_topic_analysis(row[TEXT_COLUMN]))
        sentiment = row[SENTIMENT_COLUMN]

        for token in tokens:
            if token not in scores:
                scores[token] = {label: 0 for label in VALID_SENTIMENTS}
                scores[token]["count"] = 0

            scores[token][sentiment] += 1
            scores[token]["count"] += 1

    association_scores = {}

    for token, counts in scores.items():
        total = counts["count"]
        deviations = []

        for label in VALID_SENTIMENTS:
            term_rate = counts[label] / total
            deviations.append(abs(term_rate - float(baseline[label])))

        association_scores[token] = max(deviations)

    return association_scores


def build_bigram_boost_lookup(bigram_scores_df):
    """
    For each (topic, term), take the maximum custom bigram score among bigrams
    that contain the term.
    """
    lookup = {}

    for topic, topic_bigrams in bigram_scores_df.groupby("topic"):
        term_boost = {}

        for _, row in topic_bigrams.iterrows():
            bigram = row["bigram"]
            score = row["custom_bigram_score"]

            for token in bigram.split():
                term_boost[token] = max(term_boost.get(token, 0.0), score)

        lookup[topic] = term_boost

    return lookup


def build_term_score_table(df, freq_df, tfidf_df, bigram_scores_df):
    """Assemble and normalise all term-level scoring components."""
    freq_df = freq_df[freq_df["topic_term_count"] >= MIN_TERM_COUNT].reset_index(drop=True)

    positional_by_topic = {
        topic: compute_positional_scores(topic_df)
        for topic, topic_df in df.groupby("topic", sort=True)
    }
    sentiment_by_topic = {
        topic: compute_sentiment_association_scores(topic_df)
        for topic, topic_df in df.groupby("topic", sort=True)
    }
    bigram_boost_lookup = build_bigram_boost_lookup(bigram_scores_df)

    rows = []

    for _, freq_row in freq_df.iterrows():
        topic = freq_row["topic"]
        term = freq_row["term"]

        tfidf_match = tfidf_df[
            (tfidf_df["topic"] == topic) & (tfidf_df["term"] == term)
        ]

        tfidf_value = (
            float(tfidf_match["mean_tfidf_in_topic"].iloc[0])
            if len(tfidf_match) > 0
            else 0.0
        )

        rows.append({
            "topic": topic,
            "term": term,
            "freq_score": float(freq_row["topic_term_freq"]),
            "tfidf_score": tfidf_value,
            "topic_specific_score": float(freq_row["topic_specific_ratio"]),
            "sentiment_association_score": sentiment_by_topic[topic].get(term, 0.0),
            "positional_score": positional_by_topic[topic].get(term, 0.0),
            "bigram_boost_score": bigram_boost_lookup[topic].get(term, 0.0),
            "topic_term_count": int(freq_row["topic_term_count"]),
        })

    term_scores_df = pd.DataFrame(rows)

    for column in [
        "freq_score",
        "tfidf_score",
        "topic_specific_score",
        "sentiment_association_score",
        "positional_score",
        "bigram_boost_score",
    ]:
        norm_column = f"{column}_norm"
        term_scores_df[norm_column] = (
            term_scores_df
            .groupby("topic")[column]
            .transform(min_max_normalize)
            .round(6)
        )

    custom_score = np.zeros(len(term_scores_df))

    for component, weight in TERM_COMPONENT_WEIGHTS.items():
        custom_score += term_scores_df[component].values * weight

    term_scores_df["custom_term_score"] = np.round(custom_score, 6)
    term_scores_df = term_scores_df.sort_values(
        ["topic", "custom_term_score"],
        ascending=[True, False],
    ).reset_index(drop=True)

    return term_scores_df


def build_bigram_score_table(bigram_tfidf_df):
    """Build bigram custom scores from TF-IDF and distinctiveness."""
    rows = []

    for _, row in bigram_tfidf_df.iterrows():
        distinctiveness = float(row["topic_distinctiveness_ratio"])
        if np.isinf(distinctiveness):
            distinctiveness = row["topic_doc_count"]

        rows.append({
            "topic": row["topic"],
            "bigram": row["bigram"],
            "bigram_tfidf_score": float(row["mean_tfidf_in_topic"]),
            "bigram_distinctiveness": distinctiveness,
            "topic_doc_count": int(row["topic_doc_count"]),
        })

    bigram_scores_df = pd.DataFrame(rows)

    for column in ["bigram_tfidf_score", "bigram_distinctiveness"]:
        norm_column = f"{column}_norm"
        bigram_scores_df[norm_column] = (
            bigram_scores_df
            .groupby("topic")[column]
            .transform(min_max_normalize)
            .round(6)
        )

    bigram_scores_df["custom_bigram_score"] = np.round(
        0.70 * bigram_scores_df["bigram_tfidf_score_norm"]
        + 0.30 * bigram_scores_df["bigram_distinctiveness_norm"],
        6,
    )

    bigram_scores_df = bigram_scores_df.sort_values(
        ["topic", "custom_bigram_score"],
        ascending=[True, False],
    ).reset_index(drop=True)

    return bigram_scores_df


def build_term_lookup(term_scores_df):
    """Nested lookup: topic -> term -> score dictionaries."""
    lookup = {}

    for topic, topic_df in term_scores_df.groupby("topic"):
        lookup[topic] = {
            row["term"]: {
                "custom_term_score": row["custom_term_score"],
                "positional_score_norm": row["positional_score_norm"],
            }
            for _, row in topic_df.iterrows()
        }

    return lookup


def build_bigram_lookup(bigram_scores_df):
    """Nested lookup: topic -> bigram -> custom_bigram_score."""
    lookup = {}

    for topic, topic_df in bigram_scores_df.groupby("topic"):
        lookup[topic] = {
            row["bigram"]: row["custom_bigram_score"]
            for _, row in topic_df.iterrows()
        }

    return lookup


def topic_slug(topic):
    """Turn a topic label into a column-name-safe slug, e.g. 'Artificial Intelligence' -> 'artificial_intelligence'."""
    return re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")


def score_document_against_topic(tokens, bigrams, topic_terms, topic_bigrams):
    """Score one document's tokens/bigrams against a single topic's lookup tables."""
    unigram_scores = [
        topic_terms[token]["custom_term_score"]
        for token in tokens
        if token in topic_terms
    ]
    bigram_scores = [
        topic_bigrams[bigram]
        for bigram in bigrams
        if bigram in topic_bigrams
    ]
    positional_scores = [
        topic_terms[token]["positional_score_norm"] * position_weight(index)
        for index, token in enumerate(tokens)
        if token in topic_terms
    ]

    document_unigram_score = float(np.mean(unigram_scores)) if unigram_scores else 0.0
    document_bigram_score = float(np.mean(bigram_scores)) if bigram_scores else 0.0
    document_positional_score = float(np.mean(positional_scores)) if positional_scores else 0.0

    document_custom_score = (
        DOCUMENT_COMPONENT_WEIGHTS["document_unigram_score"] * document_unigram_score
        + DOCUMENT_COMPONENT_WEIGHTS["document_bigram_score"] * document_bigram_score
        + DOCUMENT_COMPONENT_WEIGHTS["document_positional_score"] * document_positional_score
    )

    return {
        "document_unigram_score": document_unigram_score,
        "document_bigram_score": document_bigram_score,
        "document_positional_score": document_positional_score,
        "document_custom_score": document_custom_score,
        "matched_unigram_count": len(unigram_scores),
        "matched_bigram_count": len(bigram_scores),
    }


def score_documents(df, term_lookup, bigram_lookup, use_true_topic=True):
    """
    Compute document-level custom scores for each post.

    use_true_topic=True (default, used for the Section 4.1-4.2 descriptive
    analysis) scores each document against its own ground-truth topic's
    lookup tables - appropriate for exploratory "how well does this post
    match its assigned topic's vocabulary" reporting. Produces one set of
    document_unigram_score / document_bigram_score / document_positional_score
    / document_custom_score columns.

    use_true_topic=False scores each document against EVERY topic's lookup
    tables independently and keeps all of them, as one set of
    document_{unigram,bigram,positional,custom}_score_<topic> columns per
    topic (4 topics x 4 components = 16 feature columns). This must be
    used whenever the output feeds a topic classifier: selecting a single
    lookup by the document's true topic label would leak the target into
    the feature (the score would only be computable if the topic were
    already known), whereas per-topic similarity scores are computable
    without knowing the label and let the classifier itself weigh which
    topic's score is most informative.

    matched_unigram_count / matched_bigram_count are always the count of
    distinct tokens/bigrams that matched ANY topic's vocabulary (union
    across topics), so they stay a single leakage-free coverage diagnostic
    in both modes.
    """
    enriched_rows = []
    topics = sorted(term_lookup.keys())

    for _, row in df.iterrows():
        tokens = tokenize_for_topic_analysis(row[TEXT_COLUMN])
        bigrams = extract_bigrams(tokens)

        record = row.to_dict()

        if use_true_topic:
            topic = row["topic"]
            topic_terms = term_lookup.get(topic, {})
            topic_bigrams = bigram_lookup.get(topic, {})
            scores = score_document_against_topic(tokens, bigrams, topic_terms, topic_bigrams)

            record.update({
                "document_unigram_score": round(scores["document_unigram_score"], 6),
                "document_bigram_score": round(scores["document_bigram_score"], 6),
                "document_positional_score": round(scores["document_positional_score"], 6),
                "document_custom_score": round(scores["document_custom_score"], 6),
                "matched_unigram_count": scores["matched_unigram_count"],
                "matched_bigram_count": scores["matched_bigram_count"],
            })
        else:
            matched_unigram_tokens = set()
            matched_bigram_tokens = set()

            for topic in topics:
                slug = topic_slug(topic)
                topic_terms = term_lookup.get(topic, {})
                topic_bigrams = bigram_lookup.get(topic, {})
                scores = score_document_against_topic(tokens, bigrams, topic_terms, topic_bigrams)

                record.update({
                    f"document_unigram_score_{slug}": round(scores["document_unigram_score"], 6),
                    f"document_bigram_score_{slug}": round(scores["document_bigram_score"], 6),
                    f"document_positional_score_{slug}": round(scores["document_positional_score"], 6),
                    f"document_custom_score_{slug}": round(scores["document_custom_score"], 6),
                })

                matched_unigram_tokens.update(token for token in tokens if token in topic_terms)
                matched_bigram_tokens.update(bigram for bigram in bigrams if bigram in topic_bigrams)

            record["matched_unigram_count"] = len(matched_unigram_tokens)
            record["matched_bigram_count"] = len(matched_bigram_tokens)

        enriched_rows.append(record)

    return pd.DataFrame(enriched_rows)


def build_weights_table():
    """Export all weights used by the scoring mechanism."""
    rows = []

    for component, weight in TERM_COMPONENT_WEIGHTS.items():
        rows.append({
            "score_level": "term",
            "component": component,
            "weight": weight,
        })

    rows.append({
        "score_level": "bigram",
        "component": "bigram_tfidf_score_norm",
        "weight": 0.70,
    })
    rows.append({
        "score_level": "bigram",
        "component": "bigram_distinctiveness_norm",
        "weight": 0.30,
    })

    for component, weight in DOCUMENT_COMPONENT_WEIGHTS.items():
        rows.append({
            "score_level": "document",
            "component": component,
            "weight": weight,
        })

    return pd.DataFrame(rows)


def build_document_summary(enriched_df):
    """Summarise document custom scores per topic."""
    summary = (
        enriched_df
        .groupby("topic", as_index=False)
        .agg(
            document_count=("post_id", "count"),
            mean_document_custom_score=("document_custom_score", "mean"),
            max_document_custom_score=("document_custom_score", "max"),
            mean_document_unigram_score=("document_unigram_score", "mean"),
            mean_document_bigram_score=("document_bigram_score", "mean"),
            mean_document_positional_score=("document_positional_score", "mean"),
        )
    )

    for column in summary.columns:
        if column != "topic" and column != "document_count":
            summary[column] = summary[column].round(6)

    return summary


def plot_top_custom_terms(term_scores_df, output_path, top_n):
    """Bar chart of highest custom_term_score terms per topic."""
    topics = sorted(term_scores_df["topic"].unique())

    fig, axes = plt.subplots(
        nrows=len(topics),
        ncols=1,
        figsize=(10, 3.5 * len(topics)),
        squeeze=False,
    )

    for ax, topic in zip(axes.flatten(), topics):
        topic_data = (
            term_scores_df[term_scores_df["topic"] == topic]
            .sort_values("custom_term_score", ascending=True)
            .tail(top_n)
        )

        color = TOPIC_COLORS.get(topic, "#333333")

        ax.barh(
            topic_data["term"],
            topic_data["custom_term_score"],
            color=color,
            alpha=0.85,
        )
        ax.set_title(f"{topic} — top {top_n} terms (custom score)")
        ax.set_xlabel("Custom term score")
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
    os.makedirs(FEATURES_DIR, exist_ok=True)

    for path, label in [
        (INPUT_PATH, "features dataset"),
        (FREQ_TABLE_PATH, "term frequency table"),
        (TFIDF_TABLE_PATH, "TF-IDF table"),
        (BIGRAM_TFIDF_TABLE_PATH, "bigram TF-IDF table"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {label}: {path}")

    df = pd.read_csv(INPUT_PATH)
    freq_df = pd.read_csv(FREQ_TABLE_PATH)
    tfidf_df = pd.read_csv(TFIDF_TABLE_PATH)
    bigram_tfidf_df = pd.read_csv(BIGRAM_TFIDF_TABLE_PATH)

    required_columns = [TEXT_COLUMN, "topic", SENTIMENT_COLUMN, "post_id"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=[TEXT_COLUMN, "topic"]).reset_index(drop=True)
    df = df[df[TEXT_COLUMN].str.strip() != ""].reset_index(drop=True)

    print(f"Loaded {len(df)} records")
    print()

    bigram_scores_df = build_bigram_score_table(bigram_tfidf_df)
    term_scores_df = build_term_score_table(
        df, freq_df, tfidf_df, bigram_scores_df
    )

    term_lookup = build_term_lookup(term_scores_df)
    bigram_lookup = build_bigram_lookup(bigram_scores_df)
    enriched_df = score_documents(df, term_lookup, bigram_lookup)

    weights_df = build_weights_table()
    document_summary_df = build_document_summary(enriched_df)

    term_scores_df.to_csv(TERM_SCORES_OUTPUT, index=False)
    bigram_scores_df.to_csv(BIGRAM_SCORES_OUTPUT, index=False)
    weights_df.to_csv(WEIGHTS_OUTPUT, index=False)
    document_summary_df.to_csv(DOC_SUMMARY_OUTPUT, index=False)
    enriched_df.to_csv(ENRICHED_OUTPUT, index=False)

    top_terms_df = (
        term_scores_df
        .groupby("topic", group_keys=False)
        .head(TOP_N_FIGURE)
        .reset_index(drop=True)
    )
    plot_top_custom_terms(top_terms_df, FIGURE_OUTPUT, TOP_N_FIGURE)

    print("Top 10 custom-scored terms per topic:")
    print()
    for topic in sorted(df["topic"].unique()):
        print(f"--- {topic} ---")
        topic_top = term_scores_df[term_scores_df["topic"] == topic].head(10)
        for _, row in topic_top.iterrows():
            print(
                f"  {row['term']:20s} "
                f"custom={row['custom_term_score']:.4f}"
            )
        print()

    print("Document score summary:")
    print(document_summary_df.to_string(index=False))
    print()

    print("Saved:")
    print(TERM_SCORES_OUTPUT)
    print(BIGRAM_SCORES_OUTPUT)
    print(WEIGHTS_OUTPUT)
    print(DOC_SUMMARY_OUTPUT)
    print(ENRICHED_OUTPUT)
    print(FIGURE_OUTPUT)


if __name__ == "__main__":
    main()
