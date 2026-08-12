"""
sentiment_unigram_bigram_performance_comparison.py

Further Analysis helper script for Section 8.3.

Purpose:
- Extract frequent sentiment-related bigrams.
- Produce both raw and filtered bigram outputs.
- Compare sentiment classification performance using:
  1. TF-IDF unigrams only
  2. TF-IDF bigrams only
  3. TF-IDF unigrams + bigrams
- Use the best-performing sentiment model identified earlier:
  weighted Multinomial Naive Bayes.

Methodological consistency:
- Same final resolved sentiment dataset.
- Same target variable: final_sentiment.
- Same train/test split logic as Section 7:
  test_size = 0.20, random_state = 42, stratification by final_sentiment.
- Same main metric logic:
  macro-F1 is treated as the key comparison metric.

Expected input:
- A CSV file with a cleaned/modeling text column and a final_sentiment column.
- The script tries to find the dataset automatically from common project paths.
- If automatic detection fails, pass --input manually.

Outputs:
Raw bigram outputs:
- results/tables/section_7/sentiment_frequent_bigrams_top30.csv
- results/tables/section_7/sentiment_frequent_bigrams_by_class_top15.csv

Filtered bigram outputs for report use:
- results/tables/section_7/sentiment_frequent_bigrams_top30_filtered.csv
- results/tables/section_7/sentiment_frequent_bigrams_by_class_top15_filtered.csv

Performance outputs:
- results/tables/section_7/sentiment_unigram_bigram_performance_comparison.csv
- results/tables/section_7/sentiment_unigram_bigram_class_metrics.csv
- results/tables/section_7/sentiment_unigram_bigram_confusion_matrices.csv
- results/figures/section_7/sentiment_unigram_bigram_macro_f1_comparison.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.class_weight import compute_sample_weight


RANDOM_STATE = 42
TEST_SIZE = 0.20

LABEL_ORDER = ["negative", "neutral", "positive"]

REPRESENTATIONS: Dict[str, Tuple[int, int]] = {
    "unigrams_only": (1, 1),
    "bigrams_only": (2, 2),
    "unigrams_plus_bigrams": (1, 2),
}


# Tokens that should not appear in interpretive/report bigram tables.
# Classification performance is NOT affected by this filtering.
# Filtering is used only for cleaner presentation/interpretation of frequent bigrams.
FILTER_TOKENS = {
    # HTML / cleaning artifacts
    "href",
    "rel",
    "nofollow",
    "http",
    "https",
    "www",
    "com",
    "amp",
    "nbsp",

    # very common stopword-like tokens
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "am",
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "and",
    "or",
    "if",
    "it",
    "this",
    "that",
    "they",
    "you",
    "we",
    "he",
    "she",
    "as",
    "by",
    "from",
    "at",
}


# Extra exact bigrams to remove from interpretive/report tables.
FILTER_BIGRAMS_EXACT = {
    "this is",
    "in the",
    "of the",
    "to be",
    "is the",
    "on the",
    "if you",
    "it is",
    "is it",
    "you re",
    "they are",
    "how does",
    "are not",
    "is not",
    "would be",
    "for the",
    "and the",
    "from the",
    "with the",
    "at the",
    "by the",
    "as the",
    "that is",
    "there is",
    "there are",
}


def find_project_root() -> Path:
    """
    Find project root by walking upward from this script location.

    The expected project root contains folders such as:
    - src
    - results
    - data
    """
    current = Path(__file__).resolve().parent

    for parent in [current] + list(current.parents):
        if (parent / "src").exists() or (parent / "results").exists():
            return parent

    return Path.cwd()


def find_input_file(project_root: Path, user_input: str | None) -> Path:
    """
    Find the modeling dataset.

    If --input is provided, that file is used.
    Otherwise, common project paths are checked.
    """
    if user_input:
        input_path = Path(user_input)

        if not input_path.is_absolute():
            input_path = project_root / input_path

        if input_path.exists():
            return input_path

        raise FileNotFoundError(f"Input file not found: {input_path}")

    candidate_paths = [
        project_root / "data" / "modeling" / "hackernews_modeling_dataset.csv",
        project_root / "data" / "modeling" / "hackernews_posts_modeling.csv",
        project_root / "data" / "processed" / "hackernews_modeling_dataset.csv",
        project_root / "data" / "clean" / "hackernews_posts_clean.csv",
        project_root / "data" / "hackernews_modeling_dataset.csv",
    ]

    for path in candidate_paths:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find the input dataset automatically. "
        "Run the script with --input path/to/your_dataset.csv"
    )


def choose_text_column(df: pd.DataFrame, requested: str | None) -> str:
    """
    Choose the text column.

    Priority:
    1. User-provided column through --text-column
    2. text_clean
    3. text
    4. text_raw
    5. text_preview
    """
    if requested:
        if requested not in df.columns:
            raise ValueError(
                f"Requested text column '{requested}' was not found. "
                f"Available columns: {list(df.columns)}"
            )
        return requested

    candidate_columns = ["text_clean", "text", "text_raw", "text_preview"]

    for column in candidate_columns:
        if column in df.columns:
            return column

    raise ValueError(
        "No suitable text column found. Expected one of: "
        "text_clean, text, text_raw, text_preview."
    )


def load_data(
    input_path: Path,
    text_column: str | None,
    target_column: str,
) -> Tuple[pd.DataFrame, str]:
    """
    Load dataset and keep only valid sentiment labels.
    """
    df = pd.read_csv(input_path)

    selected_text_column = choose_text_column(df, text_column)

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found. "
            f"Available columns: {list(df.columns)}"
        )

    df[selected_text_column] = df[selected_text_column].fillna("").astype(str)

    df[target_column] = (
        df[target_column]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    valid_labels = set(LABEL_ORDER)

    before_rows = len(df)
    df = df[df[target_column].isin(valid_labels)].copy()
    after_rows = len(df)

    if after_rows < before_rows:
        print(
            f"[WARN] Dropped {before_rows - after_rows} rows "
            "with missing or invalid sentiment labels."
        )

    if df.empty:
        raise ValueError("No valid rows left after filtering final_sentiment labels.")

    return df, selected_text_column


def is_informative_bigram(bigram: str) -> bool:
    """
    Decide whether a bigram should be kept in interpretive/report tables.

    Important:
    - Raw bigram outputs are still saved separately.
    - Classification performance is not affected by this filtering.
    """
    normalized = str(bigram).lower().strip()
    tokens = normalized.split()

    if len(tokens) != 2:
        return False

    if normalized in FILTER_BIGRAMS_EXACT:
        return False

    if any(token in FILTER_TOKENS for token in tokens):
        return False

    if any(len(token) <= 1 for token in tokens):
        return False

    return True


def filter_bigram_table(
    bigram_df: pd.DataFrame,
    top_n: int,
    rank_column: str = "rank",
) -> pd.DataFrame:
    """
    Filter a raw bigram table and recalculate ranks.
    """
    if bigram_df.empty:
        return bigram_df.copy()

    filtered = bigram_df[bigram_df["bigram"].apply(is_informative_bigram)].copy()

    filtered = filtered.sort_values(
        by=["frequency", "bigram"],
        ascending=[False, True],
    ).head(top_n)

    if rank_column in filtered.columns:
        filtered = filtered.drop(columns=[rank_column])

    filtered.insert(0, rank_column, range(1, len(filtered) + 1))

    return filtered


def filter_bigram_table_by_class(
    bigram_df: pd.DataFrame,
    top_n_per_class: int,
) -> pd.DataFrame:
    """
    Filter a per-class bigram table and recalculate ranks within each sentiment class.
    """
    if bigram_df.empty:
        return bigram_df.copy()

    filtered = bigram_df[bigram_df["bigram"].apply(is_informative_bigram)].copy()

    output_parts = []

    for label in LABEL_ORDER:
        subset = filtered[filtered["sentiment"] == label].copy()

        if subset.empty:
            continue

        subset = subset.sort_values(
            by=["frequency", "bigram"],
            ascending=[False, True],
        ).head(top_n_per_class)

        if "rank_within_sentiment" in subset.columns:
            subset = subset.drop(columns=["rank_within_sentiment"])

        subset.insert(1, "rank_within_sentiment", range(1, len(subset) + 1))

        output_parts.append(subset)

    if not output_parts:
        return pd.DataFrame(
            columns=[
                "sentiment",
                "rank_within_sentiment",
                "bigram",
                "frequency",
            ]
        )

    return pd.concat(output_parts, ignore_index=True)


def extract_frequent_bigrams(
    texts: pd.Series,
    min_df: int,
    top_n: int,
) -> pd.DataFrame:
    """
    Extract the most frequent bigrams from the whole dataset.
    """
    vectorizer = CountVectorizer(
        ngram_range=(2, 2),
        min_df=min_df,
        lowercase=False,
    )

    matrix = vectorizer.fit_transform(texts)

    counts = np.asarray(matrix.sum(axis=0)).ravel()
    bigrams = np.array(vectorizer.get_feature_names_out())

    result = pd.DataFrame(
        {
            "bigram": bigrams,
            "frequency": counts,
        }
    )

    result = result.sort_values(
        by=["frequency", "bigram"],
        ascending=[False, True],
    ).head(top_n)

    result.insert(0, "rank", range(1, len(result) + 1))

    return result


def extract_frequent_bigrams_by_sentiment(
    df: pd.DataFrame,
    text_column: str,
    target_column: str,
    min_df: int,
    top_n_per_class: int,
) -> pd.DataFrame:
    """
    Extract the most frequent bigrams separately for each sentiment class.
    """
    output_rows = []

    for label in LABEL_ORDER:
        subset = df[df[target_column] == label].copy()

        if subset.empty:
            continue

        try:
            vectorizer = CountVectorizer(
                ngram_range=(2, 2),
                min_df=min_df,
                lowercase=False,
            )

            matrix = vectorizer.fit_transform(subset[text_column])

        except ValueError:
            continue

        counts = np.asarray(matrix.sum(axis=0)).ravel()
        bigrams = np.array(vectorizer.get_feature_names_out())

        result = pd.DataFrame(
            {
                "sentiment": label,
                "bigram": bigrams,
                "frequency": counts,
            }
        )

        result = result.sort_values(
            by=["frequency", "bigram"],
            ascending=[False, True],
        ).head(top_n_per_class)

        result.insert(1, "rank_within_sentiment", range(1, len(result) + 1))

        output_rows.append(result)

    if not output_rows:
        return pd.DataFrame(
            columns=[
                "sentiment",
                "rank_within_sentiment",
                "bigram",
                "frequency",
            ]
        )

    return pd.concat(output_rows, ignore_index=True)


def evaluate_representation(
    representation_name: str,
    ngram_range: Tuple[int, int],
    x_train_text: pd.Series,
    x_test_text: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
    min_df: int,
    max_df: float,
) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    Evaluate one TF-IDF representation with weighted Multinomial Naive Bayes.
    """
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        lowercase=False,
    )

    x_train = vectorizer.fit_transform(x_train_text)
    x_test = vectorizer.transform(x_test_text)

    sample_weight = compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    )

    model = MultinomialNB()
    model.fit(x_train, y_train, sample_weight=sample_weight)

    y_pred = model.predict(x_test)

    summary = {
        "representation": representation_name,
        "ngram_range": str(ngram_range),
        "n_features": x_train.shape[1],
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_precision": precision_score(
            y_test,
            y_pred,
            labels=LABEL_ORDER,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            y_test,
            y_pred,
            labels=LABEL_ORDER,
            average="macro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            y_test,
            y_pred,
            labels=LABEL_ORDER,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_test,
            y_pred,
            labels=LABEL_ORDER,
            average="weighted",
            zero_division=0,
        ),
    }

    report = classification_report(
        y_test,
        y_pred,
        labels=LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )

    class_metric_rows = []

    for label in LABEL_ORDER:
        class_metric_rows.append(
            {
                "representation": representation_name,
                "class": label,
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
                "support": report[label]["support"],
            }
        )

    class_metrics = pd.DataFrame(class_metric_rows)

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=LABEL_ORDER,
    )

    confusion_rows = []

    for i, actual_label in enumerate(LABEL_ORDER):
        for j, predicted_label in enumerate(LABEL_ORDER):
            confusion_rows.append(
                {
                    "representation": representation_name,
                    "actual": actual_label,
                    "predicted": predicted_label,
                    "count": int(matrix[i, j]),
                }
            )

    confusion_matrix_df = pd.DataFrame(confusion_rows)

    return summary, class_metrics, confusion_matrix_df


def save_macro_f1_plot(
    performance_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """
    Save a simple bar chart for macro-F1 across representations.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_df = performance_df.copy()

    label_map = {
        "unigrams_only": "Unigrams",
        "bigrams_only": "Bigrams",
        "unigrams_plus_bigrams": "Unigrams + Bigrams",
    }

    plot_df["plot_label"] = plot_df["representation"].map(label_map)

    plt.figure(figsize=(8, 5))
    plt.bar(plot_df["plot_label"], plot_df["macro_f1"])

    plt.xlabel("Text representation")
    plt.ylabel("Macro F1-score")
    plt.title("Sentiment Classification: Unigram vs Bigram Representations")

    plt.xticks(rotation=20, ha="right")

    upper_limit = max(0.6, float(plot_df["macro_f1"].max()) + 0.05)
    plt.ylim(0, upper_limit)

    plt.tight_layout()

    output_path = figures_dir / "sentiment_unigram_bigram_macro_f1_comparison.png"

    plt.savefig(output_path, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compare TF-IDF unigram and bigram representations "
            "for sentiment classification with weighted Multinomial Naive Bayes."
        )
    )

    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Path to final modeling dataset CSV. "
            "If omitted, common project paths are checked automatically."
        ),
    )

    parser.add_argument(
        "--text-column",
        default=None,
        help=(
            "Text column to use. If omitted, the script tries "
            "text_clean, text, text_raw, text_preview."
        ),
    )

    parser.add_argument(
        "--target-column",
        default="final_sentiment",
        help="Target column. Default: final_sentiment",
    )

    parser.add_argument(
        "--min-df",
        type=int,
        default=2,
        help="Minimum document frequency for vectorizers. Default: 2",
    )

    parser.add_argument(
        "--max-df",
        type=float,
        default=0.90,
        help="Maximum document frequency proportion for TF-IDF. Default: 0.90",
    )

    parser.add_argument(
        "--top-bigrams",
        type=int,
        default=30,
        help="Number of raw frequent bigrams to export. Default: 30",
    )

    parser.add_argument(
        "--top-bigrams-filtered",
        type=int,
        default=30,
        help="Number of filtered frequent bigrams to export. Default: 30",
    )

    parser.add_argument(
        "--top-bigrams-per-class",
        type=int,
        default=15,
        help="Number of raw frequent bigrams per sentiment class. Default: 15",
    )

    parser.add_argument(
        "--top-bigrams-per-class-filtered",
        type=int,
        default=15,
        help="Number of filtered frequent bigrams per sentiment class. Default: 15",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main execution pipeline.
    """
    args = parse_args()

    project_root = find_project_root()

    tables_dir = project_root / "results" / "tables" / "section_7"
    figures_dir = project_root / "results" / "figures" / "section_7"

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    input_path = find_input_file(
        project_root=project_root,
        user_input=args.input,
    )

    df, text_column = load_data(
        input_path=input_path,
        text_column=args.text_column,
        target_column=args.target_column,
    )

    print("[INFO] Project root:", project_root)
    print("[INFO] Input file:", input_path)
    print("[INFO] Rows loaded:", len(df))
    print("[INFO] Text column:", text_column)
    print("[INFO] Target column:", args.target_column)
    print("[INFO] Class distribution:")
    print(df[args.target_column].value_counts().reindex(LABEL_ORDER, fill_value=0))

    # Raw frequent bigrams
    frequent_bigrams_raw = extract_frequent_bigrams(
        texts=df[text_column],
        min_df=args.min_df,
        top_n=args.top_bigrams,
    )

    frequent_bigrams_raw_path = tables_dir / "sentiment_frequent_bigrams_top30.csv"
    frequent_bigrams_raw.to_csv(frequent_bigrams_raw_path, index=False)

    # Filtered frequent bigrams for report
    frequent_bigrams_filtered = filter_bigram_table(
        bigram_df=frequent_bigrams_raw,
        top_n=args.top_bigrams_filtered,
        rank_column="rank",
    )

    frequent_bigrams_filtered_path = (
        tables_dir / "sentiment_frequent_bigrams_top30_filtered.csv"
    )
    frequent_bigrams_filtered.to_csv(frequent_bigrams_filtered_path, index=False)

    # Raw frequent bigrams per class
    frequent_bigrams_by_class_raw = extract_frequent_bigrams_by_sentiment(
        df=df,
        text_column=text_column,
        target_column=args.target_column,
        min_df=args.min_df,
        top_n_per_class=args.top_bigrams_per_class,
    )

    frequent_bigrams_by_class_raw_path = (
        tables_dir / "sentiment_frequent_bigrams_by_class_top15.csv"
    )
    frequent_bigrams_by_class_raw.to_csv(
        frequent_bigrams_by_class_raw_path,
        index=False,
    )

    # Filtered frequent bigrams per class for report
    frequent_bigrams_by_class_filtered = filter_bigram_table_by_class(
        bigram_df=frequent_bigrams_by_class_raw,
        top_n_per_class=args.top_bigrams_per_class_filtered,
    )

    frequent_bigrams_by_class_filtered_path = (
        tables_dir / "sentiment_frequent_bigrams_by_class_top15_filtered.csv"
    )
    frequent_bigrams_by_class_filtered.to_csv(
        frequent_bigrams_by_class_filtered_path,
        index=False,
    )

    x_train_text, x_test_text, y_train, y_test = train_test_split(
        df[text_column],
        df[args.target_column],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[args.target_column],
    )

    print("[INFO] Train size:", len(x_train_text))
    print("[INFO] Test size:", len(x_test_text))
    print("[INFO] Test class distribution:")
    print(y_test.value_counts().reindex(LABEL_ORDER, fill_value=0))

    summary_rows: List[dict] = []
    class_metrics_tables: List[pd.DataFrame] = []
    confusion_tables: List[pd.DataFrame] = []

    for representation_name, ngram_range in REPRESENTATIONS.items():
        print(f"[INFO] Evaluating: {representation_name} {ngram_range}")

        summary, class_metrics, confusion_matrix_df = evaluate_representation(
            representation_name=representation_name,
            ngram_range=ngram_range,
            x_train_text=x_train_text,
            x_test_text=x_test_text,
            y_train=y_train,
            y_test=y_test,
            min_df=args.min_df,
            max_df=args.max_df,
        )

        summary_rows.append(summary)
        class_metrics_tables.append(class_metrics)
        confusion_tables.append(confusion_matrix_df)

    performance_df = pd.DataFrame(summary_rows)

    representation_order = [
        "unigrams_only",
        "bigrams_only",
        "unigrams_plus_bigrams",
    ]

    performance_df["representation"] = pd.Categorical(
        performance_df["representation"],
        categories=representation_order,
        ordered=True,
    )

    performance_df = performance_df.sort_values(
        "representation"
    ).reset_index(drop=True)

    class_metrics_df = pd.concat(
        class_metrics_tables,
        ignore_index=True,
    )

    confusion_df = pd.concat(
        confusion_tables,
        ignore_index=True,
    )

    performance_path = (
        tables_dir / "sentiment_unigram_bigram_performance_comparison.csv"
    )
    class_metrics_path = tables_dir / "sentiment_unigram_bigram_class_metrics.csv"
    confusion_path = tables_dir / "sentiment_unigram_bigram_confusion_matrices.csv"

    performance_df.to_csv(performance_path, index=False)
    class_metrics_df.to_csv(class_metrics_path, index=False)
    confusion_df.to_csv(confusion_path, index=False)

    save_macro_f1_plot(
        performance_df=performance_df,
        figures_dir=figures_dir,
    )

    figure_path = figures_dir / "sentiment_unigram_bigram_macro_f1_comparison.png"

    print("[OK] Saved:", frequent_bigrams_raw_path)
    print("[OK] Saved:", frequent_bigrams_filtered_path)
    print("[OK] Saved:", frequent_bigrams_by_class_raw_path)
    print("[OK] Saved:", frequent_bigrams_by_class_filtered_path)
    print("[OK] Saved:", performance_path)
    print("[OK] Saved:", class_metrics_path)
    print("[OK] Saved:", confusion_path)
    print("[OK] Saved:", figure_path)

    print("\n[RESULT] Sentiment unigram/bigram performance comparison:")
    print(performance_df.to_string(index=False))

    print("\n[RESULT] Filtered frequent bigrams:")
    print(frequent_bigrams_filtered.to_string(index=False))


if __name__ == "__main__":
    main()