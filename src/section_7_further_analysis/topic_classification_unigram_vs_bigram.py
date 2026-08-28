"""
topic_unigram_bigram_performance_comparison.py

Further Analysis helper script for Section 8.

Purpose:
- Extract frequent topic-related bigrams.
- Produce both raw and filtered bigram outputs.
- Compare topic classification performance using:
  1. TF-IDF unigrams only
  2. TF-IDF bigrams only
  3. TF-IDF unigrams + bigrams
- Use the best-performing topic model identified earlier:
  Random Forest Classifier.

Methodological consistency:
- Same final resolved dataset.
- Target variable: topic.
- Same train/test split logic:
  test_size = 0.20, random_state = 42, stratification by topic.
- Same main metric logic:
  macro-F1 is treated as the key comparison metric.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.class_weight import compute_sample_weight
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


RANDOM_STATE = 42
TEST_SIZE = 0.20

REPRESENTATIONS: Dict[str, Tuple[int, int]] = {
    "unigrams_only": (1, 1),
    "bigrams_only": (2, 2),
    "unigrams_plus_bigrams": (1, 2),
}

# Tokens that should not appear in interpretive/report bigram tables.
FILTER_TOKENS = {
    "href", "rel", "nofollow", "http", "https", "www", "com", "amp", "nbsp",
    "is", "are", "was", "were", "be", "been", "being", "am", "the", "a", "an",
    "of", "to", "in", "on", "for", "with", "and", "or", "if", "it", "this",
    "that", "they", "you", "we", "he", "she", "as", "by", "from", "at",
}

# Extra exact bigrams to remove from interpretive/report tables.
FILTER_BIGRAMS_EXACT = {
    "this is", "in the", "of the", "to be", "is the", "on the", "if you",
    "it is", "is it", "you re", "they are", "how does", "are not", "is not",
    "would be", "for the", "and the", "from the", "with the", "at the",
    "by the", "as the", "that is", "there is", "there are",
}


def find_project_root() -> Path:
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "src").exists() or (parent / "results").exists():
            return parent
    return Path.cwd()


def find_input_file(project_root: Path, user_input: str | None) -> Path:
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
        project_root / "data" / "hackernews_topic_features_dataset.csv",
        project_root / "data" / "features" / "hackernews_topic_features_dataset.csv",
    ]

    for path in candidate_paths:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find the input dataset automatically. "
        "Run the script with --input path/to/your_dataset.csv"
    )


def choose_text_column(df: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Requested text column '{requested}' not found.")
        return requested

    candidate_columns = ["text_no_stopwords", "text_clean", "text", "text_raw"]
    for column in candidate_columns:
        if column in df.columns:
            return column

    raise ValueError("No suitable text column found.")


def load_data(input_path: Path, text_column: str | None, target_column: str) -> Tuple[pd.DataFrame, str, List[str]]:
    df = pd.read_csv(input_path)
    selected_text_column = choose_text_column(df, text_column)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    df[selected_text_column] = df[selected_text_column].fillna("").astype(str)
    df[target_column] = df[target_column].fillna("").astype(str).str.strip()

    # Drop empty targets
    df = df[df[target_column] != ""].copy()
    
    label_order = sorted(df[target_column].unique().tolist())
    return df, selected_text_column, label_order


def is_informative_bigram(bigram: str) -> bool:
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


def filter_bigram_table(bigram_df: pd.DataFrame, top_n: int, rank_column: str = "rank") -> pd.DataFrame:
    if bigram_df.empty:
        return bigram_df.copy()

    filtered = bigram_df[bigram_df["bigram"].apply(is_informative_bigram)].copy()
    filtered = filtered.sort_values(by=["frequency", "bigram"], ascending=[False, True]).head(top_n)

    if rank_column in filtered.columns:
        filtered = filtered.drop(columns=[rank_column])
    filtered.insert(0, rank_column, range(1, len(filtered) + 1))

    return filtered


def filter_bigram_table_by_class(bigram_df: pd.DataFrame, top_n_per_class: int, label_order: List[str]) -> pd.DataFrame:
    if bigram_df.empty:
        return bigram_df.copy()

    filtered = bigram_df[bigram_df["bigram"].apply(is_informative_bigram)].copy()
    output_parts = []

    for label in label_order:
        subset = filtered[filtered["topic"] == label].copy()
        if subset.empty:
            continue

        subset = subset.sort_values(by=["frequency", "bigram"], ascending=[False, True]).head(top_n_per_class)
        if "rank_within_topic" in subset.columns:
            subset = subset.drop(columns=["rank_within_topic"])

        subset.insert(1, "rank_within_topic", range(1, len(subset) + 1))
        output_parts.append(subset)

    if not output_parts:
        return pd.DataFrame(columns=["topic", "rank_within_topic", "bigram", "frequency"])

    return pd.concat(output_parts, ignore_index=True)


def extract_frequent_bigrams(texts: pd.Series, min_df: int, top_n: int) -> pd.DataFrame:
    vectorizer = CountVectorizer(ngram_range=(2, 2), min_df=min_df, lowercase=True)
    matrix = vectorizer.fit_transform(texts)
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    bigrams = np.array(vectorizer.get_feature_names_out())

    result = pd.DataFrame({"bigram": bigrams, "frequency": counts})
    result = result.sort_values(by=["frequency", "bigram"], ascending=[False, True]).head(top_n)
    result.insert(0, "rank", range(1, len(result) + 1))
    return result


def extract_frequent_bigrams_by_class(df: pd.DataFrame, text_column: str, target_column: str, min_df: int, top_n_per_class: int, label_order: List[str]) -> pd.DataFrame:
    output_rows = []

    for label in label_order:
        subset = df[df[target_column] == label].copy()
        if subset.empty:
            continue

        try:
            vectorizer = CountVectorizer(ngram_range=(2, 2), min_df=min_df, lowercase=True)
            matrix = vectorizer.fit_transform(subset[text_column])
        except ValueError:
            continue

        counts = np.asarray(matrix.sum(axis=0)).ravel()
        bigrams = np.array(vectorizer.get_feature_names_out())

        result = pd.DataFrame({"topic": label, "bigram": bigrams, "frequency": counts})
        result = result.sort_values(by=["frequency", "bigram"], ascending=[False, True]).head(top_n_per_class)
        result.insert(1, "rank_within_topic", range(1, len(result) + 1))
        output_rows.append(result)

    if not output_rows:
        return pd.DataFrame(columns=["topic", "rank_within_topic", "bigram", "frequency"])

    return pd.concat(output_rows, ignore_index=True)


def evaluate_representation(
    representation_name: str, ngram_range: Tuple[int, int],
    x_train_text: pd.Series, x_test_text: pd.Series,
    y_train: pd.Series, y_test: pd.Series,
    min_df: int, max_df: float, label_order: List[str]
) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, max_df=max_df, lowercase=True)
    x_train = vectorizer.fit_transform(x_train_text)
    x_test = vectorizer.transform(x_test_text)

    # Υπολογισμός βαρών για τη σωστή στάθμιση (weighted)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    # Εκπαίδευση με Weighted Naive Bayes
    model = MultinomialNB()
    model.fit(x_train, y_train, sample_weight=sample_weight)
    y_pred = model.predict(x_test)

    summary = {
        "representation": representation_name,
        "ngram_range": str(ngram_range),
        "n_features": x_train.shape[1],
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_precision": precision_score(y_test, y_pred, labels=label_order, average="macro", zero_division=0),
        "macro_recall": recall_score(y_test, y_pred, labels=label_order, average="macro", zero_division=0),
        "macro_f1": f1_score(y_test, y_pred, labels=label_order, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test, y_pred, labels=label_order, average="weighted", zero_division=0),
    }

    report = classification_report(y_test, y_pred, labels=label_order, output_dict=True, zero_division=0)
    class_metric_rows = []

    for label in label_order:
        class_metric_rows.append({
            "representation": representation_name,
            "class": label,
            "precision": report[label]["precision"],
            "recall": report[label]["recall"],
            "f1": report[label]["f1-score"],
            "support": report[label]["support"],
        })

    class_metrics = pd.DataFrame(class_metric_rows)
    matrix = confusion_matrix(y_test, y_pred, labels=label_order)
    confusion_rows = []

    for i, actual_label in enumerate(label_order):
        for j, predicted_label in enumerate(label_order):
            confusion_rows.append({
                "representation": representation_name,
                "actual": actual_label,
                "predicted": predicted_label,
                "count": int(matrix[i, j]),
            })

    return summary, class_metrics, pd.DataFrame(confusion_rows)


def save_macro_f1_plot(performance_df: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_df = performance_df.copy()

    label_map = {
        "unigrams_only": "Unigrams",
        "bigrams_only": "Bigrams",
        "unigrams_plus_bigrams": "Unigrams + Bigrams",
    }
    plot_df["plot_label"] = plot_df["representation"].map(label_map)

    plt.figure(figsize=(8, 5))
    plt.bar(plot_df["plot_label"], plot_df["macro_f1"], color="#4C78A8")
    plt.xlabel("Text representation")
    plt.ylabel("Macro F1-score")
    plt.title("Topic Classification: Unigram vs Bigram Representations")
    plt.xticks(rotation=20, ha="right")

    upper_limit = max(0.6, float(plot_df["macro_f1"].max()) + 0.05)
    plt.ylim(0, upper_limit)
    plt.tight_layout()

    plt.savefig(figures_dir / "topic_unigram_bigram_macro_f1_comparison.png", dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare TF-IDF unigram and bigram representations for topic classification with Random Forest.")
    parser.add_argument("--input", default=None, help="Path to modeling dataset CSV.")
    parser.add_argument("--text-column", default=None, help="Text column to use.")
    parser.add_argument("--target-column", default="topic", help="Target column. Default: topic")
    parser.add_argument("--min-df", type=int, default=2, help="Minimum document frequency. Default: 2")
    parser.add_argument("--max-df", type=float, default=0.90, help="Maximum document frequency proportion. Default: 0.90")
    parser.add_argument("--top-bigrams", type=int, default=30)
    parser.add_argument("--top-bigrams-filtered", type=int, default=30)
    parser.add_argument("--top-bigrams-per-class", type=int, default=15)
    parser.add_argument("--top-bigrams-per-class-filtered", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = find_project_root()
    tables_dir = project_root / "results" / "tables" / "section_7"
    figures_dir = project_root / "results" / "figures" / "section_7"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    input_path = find_input_file(project_root, args.input)
    df, text_column, label_order = load_data(input_path, args.text_column, args.target_column)

    print(f"[INFO] Project root: {project_root}")
    print(f"[INFO] Classes found: {label_order}")

    frequent_bigrams_raw = extract_frequent_bigrams(df[text_column], args.min_df, args.top_bigrams)
    frequent_bigrams_raw.to_csv(tables_dir / "topic_frequent_bigrams_top30.csv", index=False)

    frequent_bigrams_filtered = filter_bigram_table(frequent_bigrams_raw, args.top_bigrams_filtered)
    frequent_bigrams_filtered.to_csv(tables_dir / "topic_frequent_bigrams_top30_filtered.csv", index=False)

    frequent_bigrams_by_class_raw = extract_frequent_bigrams_by_class(df, text_column, args.target_column, args.min_df, args.top_bigrams_per_class, label_order)
    frequent_bigrams_by_class_raw.to_csv(tables_dir / "topic_frequent_bigrams_by_class_top15.csv", index=False)

    frequent_bigrams_by_class_filtered = filter_bigram_table_by_class(frequent_bigrams_by_class_raw, args.top_bigrams_per_class_filtered, label_order)
    frequent_bigrams_by_class_filtered.to_csv(tables_dir / "topic_frequent_bigrams_by_class_top15_filtered.csv", index=False)

    x_train_text, x_test_text, y_train, y_test = train_test_split(
        df[text_column], df[args.target_column], test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[args.target_column]
    )

    summary_rows, class_metrics_tables, confusion_tables = [], [], []

    for representation_name, ngram_range in REPRESENTATIONS.items():
        print(f"[INFO] Evaluating: {representation_name} {ngram_range}")
        summary, class_metrics, confusion_matrix_df = evaluate_representation(
            representation_name, ngram_range, x_train_text, x_test_text, y_train, y_test, args.min_df, args.max_df, label_order
        )
        summary_rows.append(summary)
        class_metrics_tables.append(class_metrics)
        confusion_tables.append(confusion_matrix_df)

    performance_df = pd.DataFrame(summary_rows)
    performance_df["representation"] = pd.Categorical(performance_df["representation"], categories=list(REPRESENTATIONS.keys()), ordered=True)
    performance_df = performance_df.sort_values("representation").reset_index(drop=True)

    class_metrics_df = pd.concat(class_metrics_tables, ignore_index=True)
    confusion_df = pd.concat(confusion_tables, ignore_index=True)

    performance_df.to_csv(tables_dir / "topic_unigram_bigram_performance_comparison.csv", index=False)
    class_metrics_df.to_csv(tables_dir / "topic_unigram_bigram_class_metrics.csv", index=False)
    confusion_df.to_csv(tables_dir / "topic_unigram_bigram_confusion_matrices.csv", index=False)

    save_macro_f1_plot(performance_df, figures_dir)

    print("\n[RESULT] Topic unigram/bigram performance comparison:")
    print(performance_df.to_string(index=False))
    print("\n[RESULT] Filtered frequent bigrams (Top 10 preview):")
    print(frequent_bigrams_filtered.head(10).to_string(index=False))

if __name__ == "__main__":
    main()