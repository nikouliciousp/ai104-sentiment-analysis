"""
Section 6c - Sentiment Classification Evaluation

Focused evaluation of the best-performing sentiment model identified in
Section 6b: Naive Bayes with class weighting, TF-IDF representation.

Section 6b already produced the full comparison matrix. This script does not
repeat it. It re-fits only the winning model under the agreed split and
produces the artefacts required by Section 6c:

    * the confusion matrix of the best model, as a heatmap
    * the full classification report (precision, recall, F1 per class)

The split parameters are identical to Section 6b so that the evaluated model
is exactly the one reported there.

Input : data/modeling/sentiment_data.csv        (preferred)
        or data/modeling/hackernews_modeling_dataset.csv (fallback)
Output: results/tables/section_6/sentiment_confusion_matrix.csv
        results/tables/section_6/sentiment_best_model_report.csv
        results/figures/section_6/sentiment_confusion_matrix.png
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
)

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PRIMARY_INPUT = os.path.join(BASE_DIR, "data", "modeling", "sentiment_data.csv")
FALLBACK_INPUT = os.path.join(BASE_DIR, "data", "modeling", "hackernews_modeling_dataset.csv")

TABLES_DIR = os.path.join(BASE_DIR, "results", "tables", "section_6")
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures", "section_6")

# Split parameters, identical to Section 6b.
TEST_SIZE = 0.2
RANDOM_STATE = 42

# The best model, as identified in Section 6b.
LABELS = ["negative", "neutral", "positive"]
LABELS_GR = {"negative": "Αρνητικό", "neutral": "Ουδέτερο", "positive": "Θετικό"}

VECTORIZER_PARAMS = {
    "lowercase": True,
    "ngram_range": (1, 1),
    "min_df": 2,
    "max_df": 0.9,
}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_dataset():
    if os.path.exists(PRIMARY_INPUT):
        df = pd.read_csv(PRIMARY_INPUT)
        source = PRIMARY_INPUT
    else:
        df = pd.read_csv(FALLBACK_INPUT)
        source = FALLBACK_INPUT

    if "sentiment_3class" not in df.columns:
        df["sentiment_3class"] = df["final_sentiment"]

    df = df.dropna(subset=["text_clean"]).reset_index(drop=True)
    print(f"Loaded {len(df)} records from {os.path.basename(source)}")
    return df


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    df = load_dataset()

    y = df["sentiment_3class"]

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["text_clean"],
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # TF-IDF fit on training data only, then applied to the test data.
    vectorizer = TfidfVectorizer(**VECTORIZER_PARAMS)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    # Best model: Naive Bayes with class weighting via sample_weight.
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    model = MultinomialNB()
    model.fit(X_train, y_train, sample_weight=sample_weight)

    y_pred = model.predict(X_test)

    # ----------------------------------------------------------------------
    # Metrics
    # ----------------------------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", labels=LABELS)

    print("\nBest model: Naive Bayes (weighted), TF-IDF")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"macro-F1 : {macro_f1:.4f}")

    print("\nClassification report:")
    report_text = classification_report(
        y_test, y_pred, labels=LABELS, zero_division=0
    )
    print(report_text)

    # Save the report as a table
    report_dict = classification_report(
        y_test, y_pred, labels=LABELS, output_dict=True, zero_division=0
    )
    report_rows = []
    for label in LABELS:
        row = report_dict[label]
        report_rows.append({
            "class": label,
            "precision": round(row["precision"], 4),
            "recall": round(row["recall"], 4),
            "f1": round(row["f1-score"], 4),
            "support": int(row["support"]),
        })
    report_rows.append({
        "class": "macro avg",
        "precision": round(report_dict["macro avg"]["precision"], 4),
        "recall": round(report_dict["macro avg"]["recall"], 4),
        "f1": round(report_dict["macro avg"]["f1-score"], 4),
        "support": int(report_dict["macro avg"]["support"]),
    })
    pd.DataFrame(report_rows).to_csv(
        os.path.join(TABLES_DIR, "sentiment_best_model_report.csv"),
        index=False, encoding="utf-8-sig",
    )

    # ----------------------------------------------------------------------
    # Confusion matrix
    # ----------------------------------------------------------------------

    cm = confusion_matrix(y_test, y_pred, labels=LABELS)

    cm_df = pd.DataFrame(
        cm,
        index=[f"actual_{l}" for l in LABELS],
        columns=[f"pred_{l}" for l in LABELS],
    )
    cm_df.to_csv(
        os.path.join(TABLES_DIR, "sentiment_confusion_matrix.csv"),
        encoding="utf-8-sig",
    )

    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(cm_df.to_string())

    # Heatmap
    plt.figure(figsize=(7, 6))
    labels_gr = [LABELS_GR[l] for l in LABELS]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels_gr,
        yticklabels=labels_gr,
        cbar_kws={"label": "Πλήθος"},
    )
    plt.xlabel("Προβλεπόμενη κλάση")
    plt.ylabel("Πραγματική κλάση")
    plt.title("Confusion Matrix — Naive Bayes (weighted), TF-IDF")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, "sentiment_confusion_matrix.png"),
        dpi=300, bbox_inches="tight",
    )
    plt.close()

    print("\nSaved:")
    print(" ", os.path.join(TABLES_DIR, "sentiment_confusion_matrix.csv"))
    print(" ", os.path.join(TABLES_DIR, "sentiment_best_model_report.csv"))
    print(" ", os.path.join(FIGURES_DIR, "sentiment_confusion_matrix.png"))


if __name__ == "__main__":
    main()