"""
Illustrative comparison: baseline TF-IDF vs. TF-IDF + custom score features.

This is a lightweight, single-model comparison for Section 4.4 ("Compare the
custom scoring representation with a standard baseline representation...").
It reuses the leakage-safe train/test split and features from
topic_classification_features.py and fits a single MultinomialNB classifier
(one of the three classifiers required by Section 5) on each representation.

This script is NOT the Section 5 deliverable: the full benchmark across
Naive Bayes, KNN and Random Forest, with class-imbalance handling, belongs
to Section 5 Topic Classification. Its purpose here is only to give the
4.4 discussion a concrete, reproducible predictive-performance number.

Input : data/features/topic_classification_features.csv
        data/features/hackernews_topic_features_dataset.csv
Output: results/tables/section_4/topic_classification_representation_comparison.csv
        results/figures/section_4/topic_classification_representation_comparison.png
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FEATURES_PATH = os.path.join(BASE_DIR, "data", "features", "topic_classification_features.csv")
TEXT_DATASET_PATH = os.path.join(
    BASE_DIR, "data", "features", "hackernews_topic_features_dataset.csv"
)

TABLES_DIR = os.path.join(BASE_DIR, "results", "tables", "section_4")
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures", "section_4")

COMPARISON_OUTPUT = os.path.join(
    TABLES_DIR, "topic_classification_representation_comparison.csv"
)
FIGURE_OUTPUT = os.path.join(
    FIGURES_DIR, "topic_classification_representation_comparison.png"
)

TEXT_COLUMN = "text_no_stopwords"
CUSTOM_FEATURE_COLUMNS = [
    "document_custom_score",
    "document_unigram_score",
    "document_bigram_score",
    "document_positional_score",
]

# Aligned with the rest of Section 4 (topic_tfidf_analysis.py, etc.)
VECTORIZER_PARAMS = {
    "lowercase": True,
    "ngram_range": (1, 1),
    "min_df": 2,
    "max_df": 0.9,
}

METRIC_LABELS = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]


def evaluate_representation(label, X_train, X_test, y_train, y_test):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "representation": label,
        "n_features": X_train.shape[1],
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "macro_precision": round(
            precision_score(y_test, y_pred, average="macro", zero_division=0), 4
        ),
        "macro_recall": round(
            recall_score(y_test, y_pred, average="macro", zero_division=0), 4
        ),
        "macro_f1": round(
            f1_score(y_test, y_pred, average="macro", zero_division=0), 4
        ),
    }


def plot_comparison(comparison_df, output_path):
    x = np.arange(len(METRIC_LABELS))
    width = 0.35
    colors = ["#4C78A8", "#F58518"]

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (_, row) in enumerate(comparison_df.iterrows()):
        values = [row[metric] for metric in METRIC_LABELS]
        bars = ax.bar(
            x + (i - 0.5) * width, values, width,
            label=row["representation"], color=colors[i % len(colors)], alpha=0.85,
        )
        ax.bar_label(bars, labels=[f"{v:.4f}" for v in values], padding=3, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"])
    ax.set_ylim(0.90, 1.0)
    ax.set_title("Baseline vs. enriched representation (illustrative MultinomialNB, test set)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    for path, label in [
        (FEATURES_PATH, "leakage-safe classification features"),
        (TEXT_DATASET_PATH, "topic features text dataset"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {label}: {path}")

    features_df = pd.read_csv(FEATURES_PATH)
    text_df = pd.read_csv(TEXT_DATASET_PATH)[["post_id", TEXT_COLUMN]]

    df = features_df.merge(text_df, on="post_id", how="left")

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    print(f"Train: {len(train_df)}  Test: {len(test_df)}")
    print()

    vectorizer = TfidfVectorizer(**VECTORIZER_PARAMS)
    X_train_baseline = vectorizer.fit_transform(train_df[TEXT_COLUMN])
    X_test_baseline = vectorizer.transform(test_df[TEXT_COLUMN])

    y_train = train_df["topic"]
    y_test = test_df["topic"]

    custom_train = csr_matrix(train_df[CUSTOM_FEATURE_COLUMNS].values)
    custom_test = csr_matrix(test_df[CUSTOM_FEATURE_COLUMNS].values)

    X_train_enriched = hstack([X_train_baseline, custom_train]).tocsr()
    X_test_enriched = hstack([X_test_baseline, custom_test]).tocsr()

    results = [
        evaluate_representation(
            "Baseline (TF-IDF unigram)",
            X_train_baseline, X_test_baseline, y_train, y_test,
        ),
        evaluate_representation(
            "Enriched (TF-IDF + 4 custom score features)",
            X_train_enriched, X_test_enriched, y_train, y_test,
        ),
    ]

    comparison_df = pd.DataFrame(results)
    comparison_df.to_csv(COMPARISON_OUTPUT, index=False)

    plot_comparison(comparison_df, FIGURE_OUTPUT)

    print("Representation comparison (illustrative MultinomialNB, test set):")
    print(comparison_df.to_string(index=False))
    print()

    print("Saved:")
    print(COMPARISON_OUTPUT)
    print(FIGURE_OUTPUT)


if __name__ == "__main__":
    main()
