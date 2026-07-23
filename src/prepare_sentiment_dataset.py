"""
Section 6a - Sentiment Dataset Preparation

Prepares the target variable for sentiment classification and documents the
class balance decision.

This script does NOT train any model. Its purpose is to:
  1. Analyse the class distribution of the human-annotated labels
  2. Compare the four possible target definitions
  3. Produce the modelling target columns (multi-class and binary)
  4. Compute and export class weights for use in Section 6b
  5. Verify that a stratified train/test split is feasible

Input : data/modeling/hackernews_modeling_dataset.csv
Output: data/modeling/sentiment_data.csv
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_path = os.path.join(
    BASE_DIR,
    "data",
    "modeling",
    "hackernews_modeling_dataset.csv"
)

output_dir = os.path.join(BASE_DIR, "data", "modeling")

sentiment_data_output_path = os.path.join(
    output_dir,
    "sentiment_data.csv"
)

tables_dir = os.path.join(BASE_DIR, "results", "tables")
figures_dir = os.path.join(BASE_DIR, "results", "figures")

distribution_output_path = os.path.join(
    tables_dir,
    "sentiment_class_distribution.csv"
)

options_output_path = os.path.join(
    tables_dir,
    "sentiment_target_options_comparison.csv"
)

class_weights_output_path = os.path.join(
    tables_dir,
    "sentiment_class_weights.csv"
)

distribution_figure_path = os.path.join(
    figures_dir,
    "sentiment_class_distribution.png"
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

VALID_LABELS = ["negative", "neutral", "positive"]

# Split parameters agreed with Member 3 so that topic classification and
# sentiment classification are evaluated on comparable partitions.
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Display settings for the figure
SENTIMENT_COLORS = {
    "negative": "#D9534F",
    "neutral": "#AAB7C4",
    "positive": "#6BBF59"
}

SENTIMENT_LABELS_GR = {
    "negative": "Αρνητικό",
    "neutral": "Ουδέτερο",
    "positive": "Θετικό"
}


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def describe_distribution(labels):
    """
    Return a summary of a label distribution.

    Reports the imbalance ratio and the majority-class baseline accuracy.
    The baseline is the accuracy obtained by always predicting the most
    frequent class, and any model must clearly exceed it to be meaningful.
    """

    counts = labels.value_counts()
    total = len(labels)

    return {
        "records": total,
        "n_classes": len(counts),
        "imbalance_ratio": counts.max() / counts.min(),
        "baseline_accuracy": counts.max() / total,
        "minority_class": counts.idxmin(),
        "minority_count": counts.min(),
        "minority_in_test": int(round(counts.min() * TEST_SIZE)),
        "counts": counts
    }


def compute_weights(labels):
    """
    Compute balanced class weights.

    Formula used by scikit-learn:
        w_j = n / (K * n_j)

    where n is the number of samples, K the number of classes and n_j the
    number of samples in class j. Because sum(w_j * n_j) = n, every class
    contributes the same total mass to the loss function.
    """

    classes = np.array(sorted(labels.unique()))
    weights = compute_class_weight("balanced", classes=classes, y=labels)

    return dict(zip(classes, weights))


def to_binary_negative(label):
    """Collapse the three classes into negative vs non-negative."""

    return "negative" if label == "negative" else "non_negative"


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    df = pd.read_csv(input_path)

    print("Loaded modeling dataset:", len(df), "records")

    required_columns = [
        "item_id",
        "post_id",
        "text_raw",
        "text_clean",
        "created_at",
        "url",
        "topic",
        "word_count",
        "final_sentiment"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # ----------------------------------------------
    # Quality gate
    # ----------------------------------------------

    invalid = df[~df["final_sentiment"].isin(VALID_LABELS)]

    if len(invalid) > 0:
        raise ValueError(
            f"Found {len(invalid)} records with an invalid label. "
            "Resolve them before preparing the sentiment dataset."
        )

    if df["text_clean"].isna().sum() > 0:
        raise ValueError("Found missing values in text_clean.")

    if df["post_id"].duplicated().sum() > 0:
        raise ValueError("Found duplicate post_id values.")

    print("Quality gate passed: no invalid labels, no missing text, no duplicates")

    # ----------------------------------------------
    # Step 1 - Class distribution of the annotated labels
    # ----------------------------------------------

    y_multi = df["final_sentiment"]

    multi_summary = describe_distribution(y_multi)

    print()
    print("Step 1 - Class distribution (multi-class)")

    distribution_rows = []

    for label in VALID_LABELS:
        count = multi_summary["counts"][label]
        share = count / multi_summary["records"] * 100
        print(f"  {label:9s} {count:5d}  {share:6.2f}%")
        distribution_rows.append({
            "class": label,
            "count": count,
            "percentage": round(share, 2)
        })

    distribution_df = pd.DataFrame(distribution_rows)

    print()
    print("  Imbalance ratio  :", round(multi_summary["imbalance_ratio"], 2), ": 1")
    print("  Baseline accuracy:", round(multi_summary["baseline_accuracy"] * 100, 2), "%")

    # ----------------------------------------------
    # Step 2 - Compare the four possible target definitions
    # ----------------------------------------------

    # Option A: keep all three classes
    option_a = y_multi

    # Option B: drop neutral, keep only polarised posts
    option_b = y_multi[y_multi != "neutral"]

    # Option C: negative vs non-negative
    option_c = y_multi.apply(to_binary_negative)

    # Option D: positive vs non-positive
    option_d = y_multi.apply(
        lambda label: "positive" if label == "positive" else "non_positive"
    )

    options = {
        "A_multiclass_3_classes": option_a,
        "B_binary_drop_neutral": option_b,
        "C_binary_negative_vs_rest": option_c,
        "D_binary_positive_vs_rest": option_d
    }

    print()
    print("Step 2 - Comparison of target definitions")

    option_rows = []

    for name, labels in options.items():
        summary = describe_distribution(labels)
        retained = summary["records"] / len(df) * 100

        option_rows.append({
            "option": name,
            "records_kept": summary["records"],
            "records_kept_pct": round(retained, 1),
            "n_classes": summary["n_classes"],
            "imbalance_ratio": round(summary["imbalance_ratio"], 2),
            "baseline_accuracy_pct": round(summary["baseline_accuracy"] * 100, 1),
            "minority_class": summary["minority_class"],
            "minority_count": summary["minority_count"],
            "minority_in_test_set": summary["minority_in_test"]
        })

        print(
            f"  {name:28s} "
            f"kept={summary['records']:4d} ({retained:5.1f}%)  "
            f"ratio={summary['imbalance_ratio']:5.2f}:1  "
            f"baseline={summary['baseline_accuracy'] * 100:5.1f}%  "
            f"minority_test={summary['minority_in_test']:3d}"
        )

    options_df = pd.DataFrame(option_rows)

    # ----------------------------------------------
    # Step 3 - Apply the decision
    #
    # Primary analysis  : Option A (multi-class), keeps all annotated
    #                     information and avoids the closed-world assumption
    #                     that Option B would impose.
    # Secondary analysis: Option C (binary), best balanced and answers the
    #                     practical question of detecting negativity.
    # ----------------------------------------------

    df["sentiment_3class"] = y_multi
    df["sentiment_binary"] = option_c

    print()
    print("Step 3 - Target columns created")
    print("  sentiment_3class : primary target   (negative / neutral / positive)")
    print("  sentiment_binary : secondary target (negative / non_negative)")

    # ----------------------------------------------
    # Step 4 - Class weights for Section 6b
    # ----------------------------------------------

    weights_multi = compute_weights(df["sentiment_3class"])
    weights_binary = compute_weights(df["sentiment_binary"])

    print()
    print("Step 4 - Balanced class weights")

    weight_rows = []

    print("  Multi-class:")
    for label, weight in weights_multi.items():
        print(f"    {label:13s} w = {weight:.4f}")
        weight_rows.append({
            "target": "sentiment_3class",
            "class": label,
            "weight": round(weight, 4)
        })

    print("  Binary:")
    for label, weight in weights_binary.items():
        print(f"    {label:13s} w = {weight:.4f}")
        weight_rows.append({
            "target": "sentiment_binary",
            "class": label,
            "weight": round(weight, 4)
        })

    weights_df = pd.DataFrame(weight_rows)

    # Applicability of class weighting differs per classifier.
    # This is verified against the scikit-learn API and must be reported.
    print()
    print("  Applicability in Section 6b:")
    print("    RandomForestClassifier : class_weight='balanced' supported")
    print("    MultinomialNB          : no class_weight, pass sample_weight in fit()")
    print("    KNeighborsClassifier   : neither supported, document as a limitation")

    # ----------------------------------------------
    # Step 5 - Verify that a stratified split is feasible
    #
    # The split itself is performed in Section 6b. Here we only confirm that
    # every class survives stratification with the agreed parameters.
    # ----------------------------------------------

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["sentiment_3class"]
    )

    print()
    print("Step 5 - Stratified split feasibility check")
    print(f"  Parameters: test_size={TEST_SIZE}, random_state={RANDOM_STATE}, stratify=sentiment_3class")
    print(f"  Train records: {len(train_df)}")
    print(f"  Test records : {len(test_df)}")

    print("  Test set composition:")
    for label in VALID_LABELS:
        count = (test_df["sentiment_3class"] == label).sum()
        print(f"    {label:9s} {count:4d}")

    smallest_test_class = test_df["sentiment_3class"].value_counts().min()

    if smallest_test_class < 10:
        print()
        print("  WARNING: smallest class in the test set has fewer than 10 samples.")
        print("  Per-class metrics for that class will have wide confidence intervals.")

    # ----------------------------------------------
    # Output columns
    # ----------------------------------------------

    sentiment_df = df[
        [
            "item_id",
            "post_id",
            "text_raw",
            "text_clean",
            "created_at",
            "url",
            "topic",
            "word_count",
            "final_sentiment",
            "sentiment_3class",
            "sentiment_binary"
        ]
    ].copy()

    # ----------------------------------------------
    # Figure
    # ----------------------------------------------

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))

    multi_counts = df["sentiment_3class"].value_counts().reindex(VALID_LABELS)

    axes[0].bar(
        [SENTIMENT_LABELS_GR[label] for label in VALID_LABELS],
        multi_counts.values,
        color=[SENTIMENT_COLORS[label] for label in VALID_LABELS]
    )
    axes[0].set_title("Πολλαπλή Κατηγοριοποίηση (3 κλάσεις)")
    axes[0].set_ylabel("Αριθμός Εγγραφών")

    for index, value in enumerate(multi_counts.values):
        axes[0].text(
            index,
            value + 8,
            f"{value}\n({value / len(df) * 100:.1f}%)",
            ha="center",
            fontsize=9
        )

    binary_counts = df["sentiment_binary"].value_counts()
    binary_order = ["negative", "non_negative"]
    binary_counts = binary_counts.reindex(binary_order)

    axes[1].bar(
        ["Αρνητικό", "Μη Αρνητικό"],
        binary_counts.values,
        color=["#D9534F", "#AAB7C4"]
    )
    axes[1].set_title("Δυαδική Κατηγοριοποίηση (σύγκριση)")
    axes[1].set_ylabel("Αριθμός Εγγραφών")

    for index, value in enumerate(binary_counts.values):
        axes[1].text(
            index,
            value + 8,
            f"{value}\n({value / len(df) * 100:.1f}%)",
            ha="center",
            fontsize=9
        )

    plt.suptitle("Κατανομή Κλάσεων Συναισθήματος (Ενότητα 6α)")
    plt.tight_layout()
    plt.savefig(distribution_figure_path, dpi=300, bbox_inches="tight")
    plt.close()

    # ----------------------------------------------
    # Save
    # ----------------------------------------------

    sentiment_df.to_csv(
        sentiment_data_output_path,
        index=False,
        encoding="utf-8-sig"
    )

    distribution_df.to_csv(
        distribution_output_path,
        index=False,
        encoding="utf-8-sig"
    )

    options_df.to_csv(
        options_output_path,
        index=False,
        encoding="utf-8-sig"
    )

    weights_df.to_csv(
        class_weights_output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("Sentiment dataset created:", len(sentiment_df), "records")

    print()
    print("Files saved:")
    print(sentiment_data_output_path)
    print(distribution_output_path)
    print(options_output_path)
    print(class_weights_output_path)
    print(distribution_figure_path)


if __name__ == "__main__":
    main()