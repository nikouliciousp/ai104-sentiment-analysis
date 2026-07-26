"""
Section 6b - Sentiment Classification

Trains and evaluates three classifiers (Naive Bayes, K-Nearest Neighbours,
Random Forest) on the sentiment-labelled Hacker News posts.

Design of the experiment
-------------------------
The evaluation is organised as a matrix:

    3 models  x  2 weighting configurations  x  2 target definitions

Weighting configurations:
    * unweighted : no class weighting on any model. This gives a fair
                   comparison between the three algorithms under identical
                   conditions.
    * weighted   : class weighting applied where the algorithm supports it.
                   Random Forest via class_weight='balanced', Naive Bayes via
                   sample_weight in fit(). K-Nearest Neighbours cannot be
                   weighted and is therefore identical to its unweighted run;
                   this is reported, not hidden.

Target definitions (produced in Section 6a):
    * multiclass : negative / neutral / positive  (primary analysis)
    * binary     : negative / non_negative        (secondary comparison)

Text representation:
    Both Bag-of-Words and TF-IDF are run so that the choice of TF-IDF can be
    justified empirically as well as theoretically.

The split parameters are fixed and shared with the topic-classification member
so that the two tasks are evaluated on comparable partitions.

Input : data/modeling/sentiment_data.csv        (preferred)
        or data/modeling/hackernews_modeling_dataset.csv (fallback)
Output: results/tables/sentiment_model_comparison.csv
        results/tables/sentiment_per_class_metrics.csv
        results/tables/sentiment_representation_comparison.csv
        results/figures/sentiment_model_comparison.png
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    classification_report,
)

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PRIMARY_INPUT = os.path.join(BASE_DIR, "data", "modeling", "sentiment_data.csv")
FALLBACK_INPUT = os.path.join(BASE_DIR, "data", "modeling", "hackernews_modeling_dataset.csv")

TABLES_DIR = os.path.join(BASE_DIR, "results", "tables")
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures")

# Split parameters. Fixed and shared with the topic-classification member.
TEST_SIZE = 0.2
RANDOM_STATE = 42

# TF-IDF / BoW vectoriser parameters. Kept identical between representations
# so that the only difference is the presence of the IDF term.
VECTORIZER_PARAMS = {
    "lowercase": True,
    "ngram_range": (1, 1),
    "min_df": 2,          # ignore terms appearing in a single document only
    "max_df": 0.9,        # ignore terms appearing in more than 90% of documents
}

KNN_NEIGHBORS = 5
RF_ESTIMATORS = 300


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_dataset():
    """Load the sentiment dataset, producing target columns if necessary."""

    if os.path.exists(PRIMARY_INPUT):
        df = pd.read_csv(PRIMARY_INPUT)
        source = PRIMARY_INPUT
    else:
        df = pd.read_csv(FALLBACK_INPUT)
        source = FALLBACK_INPUT

    # Ensure the two target columns exist
    if "sentiment_3class" not in df.columns:
        df["sentiment_3class"] = df["final_sentiment"]

    if "sentiment_binary" not in df.columns:
        df["sentiment_binary"] = df["final_sentiment"].apply(
            lambda label: "negative" if label == "negative" else "non_negative"
        )

    # The text column used for modelling
    if "text_clean" not in df.columns:
        raise ValueError("Column text_clean is required but not found.")

    df = df.dropna(subset=["text_clean"]).reset_index(drop=True)

    print(f"Loaded {len(df)} records from {os.path.basename(source)}")

    return df


# --------------------------------------------------------------------------
# Model construction
# --------------------------------------------------------------------------

def build_models(weighted):
    """
    Return the three classifiers for a given weighting configuration.

    Only Random Forest accepts class_weight directly. Naive Bayes is weighted
    through sample_weight at fit time (handled in run_experiment). KNN cannot
    be weighted at all.
    """

    rf_class_weight = "balanced" if weighted else None

    models = {
        "Naive Bayes": MultinomialNB(),
        "KNN": KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS),
        "Random Forest": RandomForestClassifier(
            n_estimators=RF_ESTIMATORS,
            random_state=RANDOM_STATE,
            class_weight=rf_class_weight,
        ),
    }

    return models


def fit_model(name, model, X_train, y_train, weighted):
    """Fit a model, applying sample_weight for Naive Bayes when weighted."""

    if weighted and name == "Naive Bayes":
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)

    return model


# --------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------

def evaluate(model, X_test, y_test, labels):
    """Compute the full set of metrics for a fitted model."""

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", labels=labels)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", labels=labels)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=labels, zero_division=0
    )

    per_class = []
    for index, label in enumerate(labels):
        per_class.append({
            "class": label,
            "precision": round(precision[index], 4),
            "recall": round(recall[index], 4),
            "f1": round(f1[index], 4),
            "support": int(support[index]),
        })

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
    }


def run_experiment(df):
    """Run the full 3 x 2 x 2 matrix plus the representation comparison."""

    targets = {
        "multiclass": ("sentiment_3class", ["negative", "neutral", "positive"]),
        "binary": ("sentiment_binary", ["negative", "non_negative"]),
    }

    comparison_rows = []
    per_class_rows = []
    representation_rows = []

    for target_name, (target_column, labels) in targets.items():

        y = df[target_column]

        # Split once per target, on the text, stratified. The same indices are
        # reused for both representations so the comparison is clean.
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            df["text_clean"],
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        for representation in ["tfidf", "bow"]:

            if representation == "tfidf":
                vectorizer = TfidfVectorizer(**VECTORIZER_PARAMS)
            else:
                vectorizer = CountVectorizer(**VECTORIZER_PARAMS)

            # Fit the vectoriser on training data only, then transform both.
            # This prevents information leakage from the test set.
            X_train = vectorizer.fit_transform(X_train_text)
            X_test = vectorizer.transform(X_test_text)

            for weighted in [False, True]:
                config = "weighted" if weighted else "unweighted"
                models = build_models(weighted)

                for name, model in models.items():
                    model = fit_model(name, model, X_train, y_train, weighted)
                    result = evaluate(model, X_test, y_test, labels)

                    row = {
                        "target": target_name,
                        "representation": representation,
                        "config": config,
                        "model": name,
                        "accuracy": result["accuracy"],
                        "macro_f1": result["macro_f1"],
                        "weighted_f1": result["weighted_f1"],
                    }
                    comparison_rows.append(row)

                    # Keep per-class metrics only for the primary configuration
                    # (multiclass, tfidf) to avoid an unwieldy table.
                    if target_name == "multiclass" and representation == "tfidf":
                        for cls in result["per_class"]:
                            per_class_rows.append({
                                "config": config,
                                "model": name,
                                **cls,
                            })

            # Representation comparison table: best macro-F1 per model,
            # unweighted, for this representation and target.
            for name in ["Naive Bayes", "KNN", "Random Forest"]:
                matching = [
                    r for r in comparison_rows
                    if r["target"] == target_name
                    and r["representation"] == representation
                    and r["config"] == "unweighted"
                    and r["model"] == name
                ]
                if matching:
                    representation_rows.append({
                        "target": target_name,
                        "model": name,
                        "representation": representation,
                        "macro_f1": matching[0]["macro_f1"],
                    })

    return comparison_rows, per_class_rows, representation_rows


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def build_figure(comparison_df):
    """Bar chart of macro-F1 for the primary target and representation."""

    primary = comparison_df[
        (comparison_df["target"] == "multiclass")
        & (comparison_df["representation"] == "tfidf")
    ]

    models = ["Naive Bayes", "KNN", "Random Forest"]
    unweighted = [
        primary[(primary["model"] == m) & (primary["config"] == "unweighted")]["macro_f1"].values[0]
        for m in models
    ]
    weighted = [
        primary[(primary["model"] == m) & (primary["config"] == "weighted")]["macro_f1"].values[0]
        for m in models
    ]

    x = np.arange(len(models))
    width = 0.36

    figure, axes = plt.subplots(figsize=(9, 5.5))

    bars_uw = axes.bar(x - width / 2, unweighted, width,
                       label="Χωρίς στάθμιση", color="#8FA9C4")
    bars_w = axes.bar(x + width / 2, weighted, width,
                      label="Με στάθμιση", color="#D9534F")

    baseline = 0.247
    axes.axhline(baseline, color="#666666", linestyle="--", linewidth=1.1)
    axes.text(len(models) - 0.5, baseline + 0.008,
              "majority baseline (0,247)", fontsize=9, color="#666666", ha="right")

    axes.set_ylabel("macro-F1")
    axes.set_title("Σύγκριση μοντέλων — macro-F1 (πολλαπλή κατηγοριοποίηση, TF-IDF)")
    axes.set_xticks(x)
    axes.set_xticklabels(models)
    axes.set_ylim(0, max(max(unweighted), max(weighted)) * 1.25)
    axes.legend()

    for bars in [bars_uw, bars_w]:
        for bar in bars:
            height = bar.get_height()
            axes.text(bar.get_x() + bar.get_width() / 2, height + 0.006,
                      f"{height:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    figure_path = os.path.join(FIGURES_DIR, "sentiment_model_comparison.png")
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()

    return figure_path


def main():
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    df = load_dataset()

    comparison_rows, per_class_rows, representation_rows = run_experiment(df)

    comparison_df = pd.DataFrame(comparison_rows)
    per_class_df = pd.DataFrame(per_class_rows)
    representation_df = pd.DataFrame(representation_rows)

    comparison_df.to_csv(
        os.path.join(TABLES_DIR, "sentiment_model_comparison.csv"),
        index=False, encoding="utf-8-sig",
    )
    per_class_df.to_csv(
        os.path.join(TABLES_DIR, "sentiment_per_class_metrics.csv"),
        index=False, encoding="utf-8-sig",
    )
    representation_df.to_csv(
        os.path.join(TABLES_DIR, "sentiment_representation_comparison.csv"),
        index=False, encoding="utf-8-sig",
    )

    figure_path = build_figure(comparison_df)

    # ----------------------------------------------------------------------
    # Console summary
    # ----------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("PRIMARY ANALYSIS: multiclass, TF-IDF")
    print("=" * 70)
    primary = comparison_df[
        (comparison_df["target"] == "multiclass")
        & (comparison_df["representation"] == "tfidf")
    ]
    print(primary[["config", "model", "accuracy", "macro_f1", "weighted_f1"]].to_string(index=False))

    print("\n" + "=" * 70)
    print("REPRESENTATION: TF-IDF vs BoW (multiclass, unweighted, macro-F1)")
    print("=" * 70)
    rep = representation_df[representation_df["target"] == "multiclass"]
    pivot = rep.pivot(index="model", columns="representation", values="macro_f1")
    print(pivot.to_string())

    print("\n" + "=" * 70)
    print("BINARY (negative vs non_negative), TF-IDF")
    print("=" * 70)
    binary = comparison_df[
        (comparison_df["target"] == "binary")
        & (comparison_df["representation"] == "tfidf")
    ]
    print(binary[["config", "model", "accuracy", "macro_f1", "weighted_f1"]].to_string(index=False))

    print("\nSaved:")
    print(" ", os.path.join(TABLES_DIR, "sentiment_model_comparison.csv"))
    print(" ", os.path.join(TABLES_DIR, "sentiment_per_class_metrics.csv"))
    print(" ", os.path.join(TABLES_DIR, "sentiment_representation_comparison.csv"))
    print(" ", figure_path)


if __name__ == "__main__":
    main()