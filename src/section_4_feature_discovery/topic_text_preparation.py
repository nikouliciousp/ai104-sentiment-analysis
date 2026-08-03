"""
Topic feature text preparation.

Builds a separate modeling dataset for topic discovery and feature scoring.
The original hackernews_modeling_dataset.csv is left unchanged so that
sentiment modelling can keep using text_clean.

Removes English stopwords (scikit-learn list) and HTML-cleaning artifact
tokens that remain in text_clean after URL stripping.

Input : data/modeling/hackernews_modeling_dataset.csv
Output: data/features/hackernews_topic_features_dataset.csv
        results/tables/section_4/topic_features_dataset_quality_summary.csv
"""

import os

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INPUT_PATH = os.path.join(
    BASE_DIR, "data", "modeling", "hackernews_modeling_dataset.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR, "data", "features", "hackernews_topic_features_dataset.csv"
)

QUALITY_SUMMARY_PATH = os.path.join(
    BASE_DIR, "results", "tables", "section_4", "topic_features_dataset_quality_summary.csv"
)

# Tokens left behind when HTML tags/attributes are stripped from raw posts.
CUSTOM_ARTIFACT_TOKENS = frozenset({
    "href",
    "rel",
    "nofollow",
    "p",
})

# Contraction fragments produced when apostrophes are removed during cleaning.
CONTRACTION_FRAGMENTS = frozenset({
    "s",
    "t",
    "re",
    "ve",
    "ll",
    "d",
    "m",
})


# --------------------------------------------------------------------------
# Stopword utilities (shared by topic discovery scripts)
# --------------------------------------------------------------------------

def load_english_stopwords():
    """Return the combined stopword set used for topic feature analysis."""
    return set(ENGLISH_STOP_WORDS) | CUSTOM_ARTIFACT_TOKENS | CONTRACTION_FRAGMENTS


def remove_stopwords(text, stopwords=None):
    """Remove stopwords and artifact tokens from a cleaned text string."""
    if stopwords is None:
        stopwords = load_english_stopwords()

    tokens = [token for token in str(text).split() if token and token not in stopwords]
    return " ".join(tokens)


def tokenize_for_topic_analysis(text):
    """Tokenize text prepared for topic discovery (no stopwords)."""
    return [token for token in str(text).split() if token]


# --------------------------------------------------------------------------
# Dataset build
# --------------------------------------------------------------------------

def build_topic_features_dataset(df):
    """Add text_no_stopwords and word_count_no_stopwords without altering text_clean."""
    stopwords = load_english_stopwords()

    output_df = df.copy()
    output_df["text_no_stopwords"] = output_df["text_clean"].apply(
        lambda text: remove_stopwords(text, stopwords)
    )
    output_df["word_count_no_stopwords"] = output_df["text_no_stopwords"].apply(
        lambda text: len(tokenize_for_topic_analysis(text))
    )

    return output_df


def build_quality_summary(df):
    """Summarise the topic-features dataset for documentation."""
    empty_text = (df["text_no_stopwords"].str.strip() == "").sum()

    summary = pd.DataFrame([
        {"metric": "total_records", "value": len(df)},
        {"metric": "missing_text_clean", "value": df["text_clean"].isna().sum()},
        {"metric": "missing_text_no_stopwords", "value": df["text_no_stopwords"].isna().sum()},
        {"metric": "empty_text_no_stopwords", "value": int(empty_text)},
        {"metric": "duplicate_post_id", "value": df["post_id"].duplicated().sum()},
        {
            "metric": "word_count_no_stopwords_min",
            "value": int(df["word_count_no_stopwords"].min()),
        },
        {
            "metric": "word_count_no_stopwords_max",
            "value": int(df["word_count_no_stopwords"].max()),
        },
        {
            "metric": "word_count_no_stopwords_mean",
            "value": round(df["word_count_no_stopwords"].mean(), 2),
        },
    ])

    return summary


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(QUALITY_SUMMARY_PATH), exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    required_columns = ["text_clean", "post_id", "topic"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=["text_clean"]).reset_index(drop=True)

    features_df = build_topic_features_dataset(df)
    quality_summary = build_quality_summary(features_df)

    features_df.to_csv(OUTPUT_PATH, index=False)
    quality_summary.to_csv(QUALITY_SUMMARY_PATH, index=False)

    print(f"Loaded {len(df)} records from {os.path.basename(INPUT_PATH)}")
    print()
    print("Stopword removal summary:")
    print(
        f"  word_count (text_clean): "
        f"{df['word_count'].min()}–{df['word_count'].max()}"
    )
    print(
        f"  word_count_no_stopwords: "
        f"{features_df['word_count_no_stopwords'].min()}"
        f"–{features_df['word_count_no_stopwords'].max()}"
    )
    print(f"  empty text_no_stopwords: {quality_summary.loc[quality_summary['metric'] == 'empty_text_no_stopwords', 'value'].iloc[0]}")
    print()
    print("Sample before / after:")
    sample = features_df.head(3)
    for _, row in sample.iterrows():
        print(f"  clean:     {row['text_clean']}")
        print(f"  no_stop:   {row['text_no_stopwords']}")
        print()

    print("Saved:")
    print(OUTPUT_PATH)
    print(QUALITY_SUMMARY_PATH)


if __name__ == "__main__":
    main()
