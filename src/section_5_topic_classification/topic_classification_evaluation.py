"""
Section 5c - Topic Classification Evaluation

Focused evaluation of the best-performing topic model identified in
Section 5b: Random Forest on the Enriched (TF-IDF + 16 custom) representation.

Section 5b already produced the full representation comparison. This script
does not repeat it. It re-fits only the winning classifier/representation
pair under the same train/test split and produces the artifacts required by
Section 5c:

    * the confusion matrix of the best model, as a heatmap
    * the full classification report (precision, recall, F1 per class)

Input : data/features/topic_classification_features.csv
        data/features/hackernews_topic_features_dataset.csv
Output: results/tables/section_5/topic_classification_confusion_matrix.csv
        results/tables/section_5/topic_classification_best_model_report.csv
        results/figures/section_5/topic_classification_confusion_matrix.png
"""

import os
from typing import cast

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

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


def load_dataset():
    features_df = pd.read_csv(FEATURES_PATH)
    text_df = pd.read_csv(TEXT_DATASET_PATH)[["post_id", TEXT_COLUMN]]

    df = features_df.merge(text_df, on="post_id", how="left")

    missing_text = df[TEXT_COLUMN].isna().sum()
    if missing_text:
        print(f"Warning: dropping {missing_text} rows with no matching text after merge")
        df = df.dropna(subset=[TEXT_COLUMN]).reset_index(drop=True)

    return df


def main():
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    df = load_dataset()

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    print(f"Train: {len(train_df)}  Test: {len(test_df)}")

    labels = sorted(df["topic"].unique())

    # Enriched representation: TF-IDF + 16 custom features.
    tfidf_vectorizer = TfidfVectorizer(**VECTORIZER_PARAMS)
    X_train_tfidf = tfidf_vectorizer.fit_transform(train_df[TEXT_COLUMN])
    X_test_tfidf = tfidf_vectorizer.transform(test_df[TEXT_COLUMN])

    custom_train = csr_matrix(train_df[CUSTOM_FEATURE_COLUMNS].values)
    custom_test = csr_matrix(test_df[CUSTOM_FEATURE_COLUMNS].values)

    X_train = hstack([X_train_tfidf, custom_train]).tocsr()
    X_test = hstack([X_test_tfidf, custom_test]).tocsr()

    y_train = train_df["topic"]
    y_test = test_df["topic"]

    # Best model: Random Forest on the Enriched representation.
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # ----------------------------------------------------------------------
    # Metrics
    # ----------------------------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", labels=labels)

    print("\nBest model: Random Forest, Enriched (TF-IDF + 16 custom)")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"macro-F1 : {macro_f1:.4f}")

    print("\nClassification report:")
    report_text = classification_report(
        y_test, y_pred, labels=labels, zero_division=0
    )
    print(report_text)

    report_dict = cast(dict, classification_report(
        y_test, y_pred, labels=labels, output_dict=True, zero_division=0
    ))
    report_rows = []
    for label in labels:
        row = report_dict[label]
        report_rows.append({
            "class": label,
            "precision": round(float(row["precision"]), 4),
            "recall": round(float(row["recall"]), 4),
            "f1": round(float(row["f1-score"]), 4),
            "support": int(row["support"]),
        })
    macro_avg = report_dict["macro avg"]
    report_rows.append({
        "class": "macro avg",
        "precision": round(float(macro_avg["precision"]), 4),
        "recall": round(float(macro_avg["recall"]), 4),
        "f1": round(float(macro_avg["f1-score"]), 4),
        "support": int(macro_avg["support"]),
    })
    pd.DataFrame(report_rows).to_csv(
        os.path.join(TABLES_DIR, "topic_classification_best_model_report.csv"),
        index=False, encoding="utf-8-sig",
    )

    # ----------------------------------------------------------------------
    # Confusion matrix
    # ----------------------------------------------------------------------

    cm = confusion_matrix(y_test, y_pred, labels=labels)

    cm_df = pd.DataFrame(
        cm,
        index=[f"actual_{l}" for l in labels],
        columns=[f"pred_{l}" for l in labels],
    )
    cm_df.to_csv(
        os.path.join(TABLES_DIR, "topic_classification_confusion_matrix.csv"),
        encoding="utf-8-sig",
    )

    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(cm_df.to_string())

    # Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Count"},
    )
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.title("Confusion Matrix — Random Forest, TF-IDF με Enriched Features")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, "topic_classification_confusion_matrix.png"),
        dpi=300, bbox_inches="tight",
    )
    plt.close()

    print("\nSaved:")
    print(" ", os.path.join(TABLES_DIR, "topic_classification_confusion_matrix.csv"))
    print(" ", os.path.join(TABLES_DIR, "topic_classification_best_model_report.csv"))
    print(" ", os.path.join(FIGURES_DIR, "topic_classification_confusion_matrix.png"))


if __name__ == "__main__":
    main()
