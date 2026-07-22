import os
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_path = os.path.join(
    BASE_DIR,
    "data",
    "annotated",
    "hackernews_annotations_resolved_final.csv"
)

tables_dir = os.path.join(BASE_DIR, "results", "tables")
figures_dir = os.path.join(BASE_DIR, "results", "figures")

counts_output_path = os.path.join(
    tables_dir,
    "topic_sentiment_counts.csv"
)

percentages_output_path = os.path.join(
    tables_dir,
    "topic_sentiment_percentages.csv"
)

stacked_bar_output_path = os.path.join(
    figures_dir,
    "topic_sentiment_stacked_bar.png"
)

percent_stacked_bar_output_path = os.path.join(
    figures_dir,
    "topic_sentiment_100_percent_stacked_bar.png"
)


# --------------------------------------------------
# Main analysis
# --------------------------------------------------

def main():
    df = pd.read_csv(input_path)

    print("Loaded resolved final dataset:", len(df), "records")

    required_columns = ["topic", "final_sentiment"]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Fixed sentiment order
    valid_labels = ["negative", "neutral", "positive"]

    # Keep only valid sentiment labels
    df = df[df["final_sentiment"].isin(valid_labels)].copy()

    print("Valid sentiment records:", len(df))

    # --------------------------------------------------
    # Count table
    # --------------------------------------------------

    counts = pd.crosstab(
        df["topic"],
        df["final_sentiment"]
    )

    # Ensure all sentiment columns exist and are in fixed order
    for label in valid_labels:
        if label not in counts.columns:
            counts[label] = 0

    counts = counts[valid_labels]
    counts["total"] = counts.sum(axis=1)

    # --------------------------------------------------
    # Percentage table
    # --------------------------------------------------

    percentages = counts[valid_labels].div(
        counts["total"],
        axis=0
    ) * 100

    percentages = percentages.round(1)
    percentages["total"] = counts["total"]

    # --------------------------------------------------
    # Clean topic labels for display
    # --------------------------------------------------

    counts_display = counts.copy()
    percentages_display = percentages.copy()

    counts_display.index = counts_display.index.astype(str).str.replace("_", " ")
    percentages_display.index = percentages_display.index.astype(str).str.replace("_", " ")

    # --------------------------------------------------
    # Save tables
    # --------------------------------------------------

    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    counts.to_csv(counts_output_path, encoding="utf-8-sig")
    percentages.to_csv(percentages_output_path, encoding="utf-8-sig")

    # --------------------------------------------------
    # Stacked bar chart: counts
    # --------------------------------------------------

    ax = counts_display[valid_labels].plot(
        kind="bar",
        stacked=True,
        figsize=(11, 6)
    )

    plt.title("Κατανομή Συναισθήματος ανά Θέμα")
    plt.xlabel("Θέμα")
    plt.ylabel("Αριθμός Εγγραφών")
    plt.xticks(rotation=30, ha="right")

    plt.legend(
        title="Συναίσθημα",
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )

    plt.tight_layout()
    plt.savefig(stacked_bar_output_path, dpi=300, bbox_inches="tight")
    plt.close()

    # --------------------------------------------------
    # 100% stacked bar chart: percentages
    # --------------------------------------------------

    ax = percentages_display[valid_labels].plot(
        kind="bar",
        stacked=True,
        figsize=(11, 6)
    )

    plt.title("Ποσοστιαία Κατανομή Συναισθήματος ανά Θέμα")
    plt.xlabel("Θέμα")
    plt.ylabel("Ποσοστό Εγγραφών")
    plt.xticks(rotation=30, ha="right")

    plt.legend(
        title="Συναίσθημα",
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )

    plt.tight_layout()
    plt.savefig(percent_stacked_bar_output_path, dpi=300, bbox_inches="tight")
    plt.close()

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print()
    print("Topic-sentiment counts:")
    print(counts.to_string())

    print()
    print("Topic-sentiment percentages:")
    print(percentages.to_string())

    print()
    print("Files saved:")
    print(counts_output_path)
    print(percentages_output_path)
    print(stacked_bar_output_path)
    print(percent_stacked_bar_output_path)


if __name__ == "__main__":
    main()