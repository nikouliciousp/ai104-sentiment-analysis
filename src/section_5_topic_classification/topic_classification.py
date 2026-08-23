import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURES_PATH = os.path.join(BASE_DIR, "data", "features", "topic_classification_features.csv")
TEXT_DATASET_PATH = os.path.join(BASE_DIR, "data", "features", "hackernews_topic_features_dataset.csv")
TEXT_COLUMN = "text_no_stopwords"
CUSTOM_FEATURE_COLUMNS = [
    "document_unigram_score_artificial_intelligence",
    "document_bigram_score_artificial_intelligence",
    "document_positional_score_artificial_intelligence",
    "document_custom_score_artificial_intelligence",
    "document_unigram_score_climate_change",
    "document_bigram_score_climate_change",
    "document_positional_score_climate_change",
    "document_custom_score_climate_change",
    "document_unigram_score_cryptocurrency",
    "document_bigram_score_cryptocurrency",
    "document_positional_score_cryptocurrency",
    "document_custom_score_cryptocurrency",
    "document_unigram_score_cybersecurity",
    "document_bigram_score_cybersecurity",
    "document_positional_score_cybersecurity",
    "document_custom_score_cybersecurity",
]
VECTORIZER_PARAMS = {
    "lowercase": True,
    "ngram_range": (1, 1),
    "min_df": 2,
    "max_df": 0.9,
}

TABLES_DIR = os.path.join(BASE_DIR, "results", "tables", "section_5")
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures", "section_5")
COMPARISON_OUTPUT = os.path.join(TABLES_DIR, "topic_classification_representation_comparison.csv")
FIGURE_OUTPUT = os.path.join(FIGURES_DIR, "topic_classification_representation_comparison.png")

METRIC_LABELS = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]


def evaluate_representation(classifier_name, model, X_train, X_test, y_train, y_test, representation_label):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "classifier": classifier_name,
        "representation": representation_label,
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
    classifiers = comparison_df["classifier"].unique()
    x = np.arange(len(METRIC_LABELS))
    width = 0.12
    colors = ["#4C78A8", "#F58518", "#72B7B2"]

    fig, axes = plt.subplots(1, len(classifiers), figsize=(18, 5))
    if len(classifiers) == 1:
        axes = [axes]

    legend_handles, legend_labels = None, None
    for ax_idx, classifier in enumerate(classifiers):
        subset = comparison_df[comparison_df["classifier"] == classifier]
        ax = axes[ax_idx]

        for i, (_, row) in enumerate(subset.iterrows()):
            values = [row[metric] for metric in METRIC_LABELS]
            bars = ax.bar(
                x + (i - 1) * width * 2, values, width * 2,
                label=row["representation"], color=colors[i % len(colors)], alpha=0.85,
            )
            ax.bar_label(bars, labels=[f"{v:.4f}" for v in values], padding=3, fontsize=6)

        ax.set_xticks(x)
        ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1"], fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_title(f"{classifier}", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        if ax_idx == 0:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    fig.suptitle("Κατηγοριοποίηση Θεμάτων: Bag-Of-Words έναντι TF-IDF με Enriched Features", fontsize=12, y=1.04)
    fig.legend(
        legend_handles, legend_labels, loc="upper center",
        bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=9, frameon=False,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    features_df = pd.read_csv(FEATURES_PATH)
    text_df = pd.read_csv(TEXT_DATASET_PATH)[["post_id", TEXT_COLUMN]]

    df = features_df.merge(text_df, on="post_id", how="left")

    missing_text = df[TEXT_COLUMN].isna().sum()
    if missing_text:
        print(f"Warning: dropping {missing_text} rows with no matching text after merge")
        df = df.dropna(subset=[TEXT_COLUMN]).reset_index(drop=True)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    print(f"Train: {len(train_df)}  Test: {len(test_df)}")
    print()

    # Bag of Words
    bow_vectorizer = CountVectorizer(**VECTORIZER_PARAMS)
    X_train_bow = bow_vectorizer.fit_transform(train_df[TEXT_COLUMN])
    X_test_bow = bow_vectorizer.transform(test_df[TEXT_COLUMN])

    # TF-IDF
    tfidf_vectorizer = TfidfVectorizer(**VECTORIZER_PARAMS)
    X_train_tfidf = tfidf_vectorizer.fit_transform(train_df[TEXT_COLUMN])
    X_test_tfidf = tfidf_vectorizer.transform(test_df[TEXT_COLUMN])

    y_train = train_df["topic"]
    y_test = test_df["topic"]

    custom_train = csr_matrix(train_df[CUSTOM_FEATURE_COLUMNS].values)
    custom_test = csr_matrix(test_df[CUSTOM_FEATURE_COLUMNS].values)

    X_train_enriched = hstack([X_train_tfidf, custom_train]).tocsr()
    X_test_enriched = hstack([X_test_tfidf, custom_test]).tocsr()

    print(f"Train: {len(train_df)}  Test: {len(test_df)}")
    print(f"BoW features: {X_train_bow.shape[1]}")
    print(f"Enriched features: {X_train_enriched.shape[1]}")
    print()

    classifiers = [
        ("Naive Bayes", MultinomialNB()),
        ("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=5)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ]

    results = []

    for classifier_name, model in classifiers:
        print(f"Evaluating {classifier_name}...")

        results.append(
            evaluate_representation(
                classifier_name, model,
                X_train_bow, X_test_bow, y_train, y_test,
                "Bag of Words"
            )
        )

        results.append(
            evaluate_representation(
                classifier_name, model,
                X_train_enriched, X_test_enriched, y_train, y_test,
                "Enriched (TF-IDF + 16 custom)"
            )
        )

    comparison_df = pd.DataFrame(results)
    comparison_df.to_csv(COMPARISON_OUTPUT, index=False)

    plot_comparison(comparison_df, FIGURE_OUTPUT)

    print()
    print("Topic Classification Results (Test Set):")
    print(comparison_df.to_string(index=False))
    print()

    print("Saved:")
    print(COMPARISON_OUTPUT)
    print(FIGURE_OUTPUT)


if __name__ == "__main__":
    main()
