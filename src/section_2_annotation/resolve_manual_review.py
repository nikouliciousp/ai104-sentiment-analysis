import os
import pandas as pd

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

final_input_path = os.path.join(
    BASE_DIR,
    "data",
    "annotated",
    "hackernews_annotations_final.csv"
)

manual_review_input_path = os.path.join(
    BASE_DIR,
    "data",
    "annotated",
    "hackernews_manual_review_completed.csv"
)

output_dir = os.path.join(BASE_DIR, "data", "annotated")

resolved_output_path = os.path.join(
    output_dir,
    "hackernews_annotations_resolved_final.csv"
)

remaining_review_output_path = os.path.join(
    output_dir,
    "hackernews_manual_review_remaining.csv"
)

summary_dir = os.path.join(BASE_DIR, "results", "tables", "section_2")

resolved_distribution_output_path = os.path.join(
    summary_dir,
    "resolved_sentiment_distribution_summary.csv"
)

resolved_quality_output_path = os.path.join(
    summary_dir,
    "resolved_annotation_quality_summary.csv"
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

VALID_LABELS = {"positive", "neutral", "negative"}


# --------------------------------------------------
# Helper function
# --------------------------------------------------

def normalize_label(label):
    """
    Normalize labels from manual review.
    Valid labels: positive, neutral, negative.
    Empty / invalid labels return None.
    """

    if pd.isna(label):
        return None

    label = str(label).strip().lower()

    if label == "":
        return None

    if label not in VALID_LABELS:
        return None

    return label


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    final_df = pd.read_csv(final_input_path)
    review_df = pd.read_csv(manual_review_input_path)

    print("Loaded final annotations:", len(final_df), "records")
    print("Loaded manual review completed:", len(review_df), "records")

    required_final_columns = [
        "item_id",
        "post_id",
        "topic",
        "created_at",
        "url",
        "text_preview",
        "annotator_1",
        "annotator_2",
        "annotator_3",
        "annotator_4",
        "final_sentiment"
    ]

    required_review_columns = [
        "post_id",
        "adjudicated_sentiment"
    ]

    missing_final_columns = [
        col for col in required_final_columns
        if col not in final_df.columns
    ]

    missing_review_columns = [
        col for col in required_review_columns
        if col not in review_df.columns
    ]

    if missing_final_columns:
        raise ValueError(f"Missing columns in final file: {missing_final_columns}")

    if missing_review_columns:
        raise ValueError(f"Missing columns in manual review file: {missing_review_columns}")

    # Normalize both final_sentiment and adjudicated_sentiment
    final_df["final_sentiment"] = final_df["final_sentiment"].astype(str).str.strip().str.lower()

    review_df["adjudicated_sentiment"] = review_df["adjudicated_sentiment"].apply(
        normalize_label
    )

    # Keep only required mapping columns
    review_mapping = review_df[
        [
            "post_id",
            "adjudicated_sentiment"
        ]
    ].copy()

    # Merge adjudicated sentiment onto final dataset by post_id
    resolved_df = final_df.merge(
        review_mapping,
        on="post_id",
        how="left"
    )

    # Replace manual_review labels where adjudicated_sentiment exists
    mask_manual_review = resolved_df["final_sentiment"] == "manual_review"
    mask_has_adjudication = resolved_df["adjudicated_sentiment"].isin(VALID_LABELS)

    resolved_df.loc[
        mask_manual_review & mask_has_adjudication,
        "final_sentiment"
    ] = resolved_df.loc[
        mask_manual_review & mask_has_adjudication,
        "adjudicated_sentiment"
    ]

    # Identify any remaining unresolved records
    unresolved_df = resolved_df[
        ~resolved_df["final_sentiment"].isin(VALID_LABELS)
    ].copy()

    # Drop helper column from final output, but keep it in unresolved file if useful
    final_output_df = resolved_df.drop(columns=["adjudicated_sentiment"])

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    # Save resolved final dataset
    final_output_df.to_csv(
        resolved_output_path,
        index=False,
        encoding="utf-8-sig"
    )

    # Save remaining unresolved records, if any
    unresolved_df.to_csv(
        remaining_review_output_path,
        index=False,
        encoding="utf-8-sig"
    )

    # Final sentiment distribution summary
    resolved_distribution = (
        final_output_df
        .groupby(["topic", "final_sentiment"])
        .size()
        .reset_index(name="count")
    )

    resolved_distribution.to_csv(
        resolved_distribution_output_path,
        index=False,
        encoding="utf-8-sig"
    )

    # Final quality summary
    quality_summary = pd.DataFrame({
        "metric": [
            "total_records",
            "resolved_final_records",
            "remaining_unresolved_records",
            "positive_records",
            "neutral_records",
            "negative_records"
        ],
        "value": [
            len(final_output_df),
            final_output_df["final_sentiment"].isin(VALID_LABELS).sum(),
            len(unresolved_df),
            (final_output_df["final_sentiment"] == "positive").sum(),
            (final_output_df["final_sentiment"] == "neutral").sum(),
            (final_output_df["final_sentiment"] == "negative").sum()
        ]
    })

    quality_summary.to_csv(
        resolved_quality_output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("Resolved final sentiment distribution:")
    print(final_output_df["final_sentiment"].value_counts().to_string())

    print()
    print("Resolved final sentiment distribution by topic:")
    print(
        pd.crosstab(
            final_output_df["topic"],
            final_output_df["final_sentiment"]
        ).to_string()
    )

    print()
    print("Resolved annotation quality:")
    print(quality_summary.to_string(index=False))

    print()
    print("Files saved:")
    print(resolved_output_path)
    print(remaining_review_output_path)
    print(resolved_distribution_output_path)
    print(resolved_quality_output_path)


if __name__ == "__main__":
    main()