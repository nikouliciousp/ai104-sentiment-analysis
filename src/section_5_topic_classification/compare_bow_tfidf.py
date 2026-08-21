import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURES_PATH = os.path.join(BASE_DIR, "data", "features", "topic_classification_features.csv")
TEXT_DATASET_PATH = os.path.join(BASE_DIR, "data", "features", "hackernews_topic_features_dataset.csv")
TEXT_COLUMN = "text_no_stopwords"
VECTORIZER_PARAMS = {
    "lowercase": True,
    "ngram_range": (1, 1),
    "min_df": 2,
    "max_df": 0.9,
}

TABLES_DIR = os.path.join(BASE_DIR, "results", "tables", "section_5")
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures", "section_5")
COMPARISON_OUTPUT = os.path.join(TABLES_DIR, "topic_classification_bow_tfidf_comparison.csv")
CV_OUTPUT = os.path.join(TABLES_DIR, "topic_classification_bow_tfidf_cv_summary.csv")
SIGNIFICANCE_OUTPUT = os.path.join(TABLES_DIR, "topic_classification_bow_tfidf_significance.csv")
FIGURE_OUTPUT = os.path.join(FIGURES_DIR, "topic_classification_bow_tfidf_comparison.png")

METRIC_LABELS = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
CV_SCORING = {
    "accuracy": "accuracy",
    "macro_precision": "precision_macro",
    "macro_recall": "recall_macro",
    "macro_f1": "f1_macro",
}
N_SPLITS = 5
RANDOM_STATE = 42

CLASSIFIERS = [
    ("Naive Bayes", MultinomialNB()),
    ("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=5)),
    ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)),
]
REPRESENTATIONS = ["Bag-Of-Words", "TF-IDF"]


def build_pipeline(representation_label, classifier_name, model):
    """
    Build a fit-safe pipeline (vectorizer fit only on training folds, no leakage).

    KNN is distance-based, so raw BoW counts (unbounded magnitude, grows with
    document length) and TF-IDF vectors (L2-normalized by default in sklearn)
    are not on a comparable scale. Without correcting for this, KNN will
    favor TF-IDF for reasons that have nothing to do with term-weighting
    quality. We L2-normalize both representations for KNN so the comparison
    isolates the weighting scheme rather than vector length.
    """
    if representation_label == "Bag-Of-Words":
        vectorizer = CountVectorizer(**VECTORIZER_PARAMS)
    else:
        vectorizer = TfidfVectorizer(**VECTORIZER_PARAMS)

    steps = [("vectorizer", vectorizer)]
    if classifier_name == "K-Nearest Neighbors":
        steps.append(("normalizer", Normalizer(norm="l2")))
    steps.append(("classifier", model))

    return Pipeline(steps)


def cross_validate_representation(classifier_name, model, X_text, y, representation_label):
    """5-fold stratified CV on the training set; returns per-fold scores."""
    pipeline = build_pipeline(representation_label, classifier_name, model)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    scores = cross_validate(
        pipeline, X_text, y, cv=cv,
        scoring=CV_SCORING, n_jobs=1,
    )

    return {metric: scores[f"test_{metric}"] for metric in METRIC_LABELS}


def summarize_cv(cv_results):
    """cv_results: dict[(classifier, representation)] -> dict[metric] -> fold array."""
    rows = []
    for (classifier_name, representation_label), metric_folds in cv_results.items():
        row = {"classifier": classifier_name, "representation": representation_label}
        for metric in METRIC_LABELS:
            folds = metric_folds[metric]
            row[f"{metric}_mean"] = round(folds.mean(), 4)
            row[f"{metric}_std"] = round(folds.std(), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def run_significance_tests(cv_results, metric="macro_f1", alpha=0.05):
    """
    Paired t-test (BoW vs TF-IDF, same CV folds) per classifier on one metric.
    Paired because both representations are evaluated on identical folds, so
    fold-to-fold difficulty is controlled for -- this is a much stronger test
    than comparing two single point estimates from one train/test split.
    """
    rows = []
    for classifier_name, _ in CLASSIFIERS:
        bow_folds = cv_results[(classifier_name, "Bag-Of-Words")][metric]
        tfidf_folds = cv_results[(classifier_name, "TF-IDF")][metric]

        t_stat, p_value = stats.ttest_rel(tfidf_folds, bow_folds)
        mean_diff = tfidf_folds.mean() - bow_folds.mean()

        if p_value < alpha:
            verdict = "TF-IDF better" if mean_diff > 0 else "BoW better"
        else:
            verdict = "no significant difference"

        rows.append({
            "classifier": classifier_name,
            "metric": metric,
            "bow_mean": round(bow_folds.mean(), 4),
            "tfidf_mean": round(tfidf_folds.mean(), 4),
            "mean_diff_tfidf_minus_bow": round(mean_diff, 4),
            "t_stat": round(t_stat, 4),
            "p_value": round(p_value, 4),
            "significant_at_0.05": p_value < alpha,
            "verdict": verdict,
        })
    return pd.DataFrame(rows)


def evaluate_on_test(classifier_name, model, X_train_text, X_test_text, y_train, y_test, representation_label):
    """Single fixed-split evaluation on the held-out test set (final numbers)."""
    pipeline = build_pipeline(representation_label, classifier_name, model)
    pipeline.fit(X_train_text, y_train)
    y_pred = pipeline.predict(X_test_text)

    n_features = pipeline.named_steps["vectorizer"].transform(X_train_text).shape[1]

    return {
        "classifier": classifier_name,
        "representation": representation_label,
        "n_features": n_features,
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
    width = 0.15
    colors = ["#4C78A8", "#F58518"]

    fig, axes = plt.subplots(1, len(classifiers), figsize=(15, 5))
    if len(classifiers) == 1:
        axes = [axes]

    legend_handles, legend_labels = None, None
    for ax_idx, classifier in enumerate(classifiers):
        subset = comparison_df[comparison_df["classifier"] == classifier]
        ax = axes[ax_idx]

        for i, (_, row) in enumerate(subset.iterrows()):
            values = [row[metric] for metric in METRIC_LABELS]
            bars = ax.bar(
                x + (i - 0.5) * width * 2, values, width * 2,
                label=row["representation"], color=colors[i % len(colors)], alpha=0.85,
            )
            ax.bar_label(bars, labels=[f"{v:.4f}" for v in values], padding=3, fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1"], fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_title(f"{classifier}", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        if ax_idx == 0:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    fig.suptitle("Κατηγοριοποίηση Θεμάτων : Bag-Of-Words έναντι TF-IDF", fontsize=12, y=1.04)
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

    x_train_text = train_df[TEXT_COLUMN]
    x_test_text = test_df[TEXT_COLUMN]
    y_train = train_df["topic"]
    y_test = test_df["topic"]

    print(f"Train: {len(train_df)}  Test: {len(test_df)}")
    print(f"Train class distribution:\n{y_train.value_counts()}")
    print()

    # ------------------------------------------------------------------
    # 1. Cross-validation on the training set: mean/std + significance.
    #    This is what tells us whether an observed BoW/TF-IDF gap is real
    #    or just noise from a single split.
    # ------------------------------------------------------------------
    cv_results = {}
    for classifier_name, model in CLASSIFIERS:
        for representation_label in REPRESENTATIONS:
            print(f"Cross-validating {classifier_name} / {representation_label}...")
            cv_results[(classifier_name, representation_label)] = cross_validate_representation(
                classifier_name, model, x_train_text, y_train, representation_label
            )

    cv_summary_df = summarize_cv(cv_results)
    cv_summary_df.to_csv(CV_OUTPUT, index=False)

    significance_df = run_significance_tests(cv_results, metric="macro_f1")
    significance_df.to_csv(SIGNIFICANCE_OUTPUT, index=False)

    print()
    print(f"{N_SPLITS}-Fold CV Summary (Training Set):")
    print(cv_summary_df.to_string(index=False))
    print()
    print("Paired significance test, BoW vs TF-IDF (macro F1, same folds):")
    print(significance_df.to_string(index=False))
    print()

    # ------------------------------------------------------------------
    # 2. Final held-out test set numbers (report these as the headline
    #    result; use the CV summary/significance table to justify how
    #    much to trust the gap between BoW and TF-IDF).
    # ------------------------------------------------------------------
    results = []

    for classifier_name, model in CLASSIFIERS:
        print(f"Evaluating {classifier_name} on held-out test set...")
        for representation_label in REPRESENTATIONS:
            results.append(
                evaluate_on_test(
                    classifier_name, model,
                    x_train_text, x_test_text, y_train, y_test,
                    representation_label,
                )
            )

    comparison_df = pd.DataFrame(results)
    comparison_df.to_csv(COMPARISON_OUTPUT, index=False)

    plot_comparison(comparison_df, FIGURE_OUTPUT)

    print()
    print("Topic Classification Results (Held-Out Test Set):")
    print(comparison_df.to_string(index=False))
    print()

    print("Saved:")
    print(COMPARISON_OUTPUT)
    print(CV_OUTPUT)
    print(SIGNIFICANCE_OUTPUT)
    print(FIGURE_OUTPUT)


if __name__ == "__main__":
    main()
