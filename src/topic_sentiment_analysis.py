import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# --------------------------------------------------
# Διαδρομές αρχείων
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

chi_square_output_path = os.path.join(
    tables_dir,
    "topic_sentiment_chi_square_test.csv"
)

expected_counts_output_path = os.path.join(
    tables_dir,
    "topic_sentiment_expected_counts.csv"
)

stacked_bar_output_path = os.path.join(
    figures_dir,
    "topic_sentiment_stacked_bar.png"
)

percent_stacked_bar_output_path = os.path.join(
    figures_dir,
    "topic_sentiment_100_percent_stacked_bar.png"
)

heatmap_output_path = os.path.join(
    figures_dir,
    "topic_sentiment_heatmap.png"
)

# --------------------------------------------------
# Ρυθμίσεις εμφάνισης
# --------------------------------------------------

VALID_LABELS = ["negative", "neutral", "positive"]

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


def rename_sentiment_columns_for_display(df):
    """
    Μετονομάζει τις στήλες sentiment μόνο για εμφάνιση στα γραφήματα.
    Δεν αλλάζει τα labels στα πραγματικά δεδομένα.
    """

    return df.rename(columns=SENTIMENT_LABELS_GR)


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

    df = df[df["final_sentiment"].isin(VALID_LABELS)].copy()

    print("Valid sentiment records:", len(df))

    # --------------------------------------------------
    # Διασταυρούμενος πίνακας topic × sentiment
    # --------------------------------------------------

    counts = pd.crosstab(
        df["topic"],
        df["final_sentiment"]
    )

    for label in VALID_LABELS:
        if label not in counts.columns:
            counts[label] = 0

    counts = counts[VALID_LABELS]
    counts["total"] = counts.sum(axis=1)

    percentages = counts[VALID_LABELS].div(
        counts["total"],
        axis=0
    ) * 100

    percentages = percentages.round(1)
    percentages["total"] = counts["total"]

    counts_display = counts.copy()
    percentages_display = percentages.copy()

    counts_display.index = counts_display.index.astype(str).str.replace("_", " ")
    percentages_display.index = percentages_display.index.astype(str).str.replace("_", " ")

    counts_for_plot = rename_sentiment_columns_for_display(
        counts_display[VALID_LABELS]
    )

    percentages_for_plot = rename_sentiment_columns_for_display(
        percentages_display[VALID_LABELS]
    )

    display_label_order = [
        SENTIMENT_LABELS_GR[label]
        for label in VALID_LABELS
    ]

    display_color_order = [
        SENTIMENT_COLORS[label]
        for label in VALID_LABELS
    ]

    counts_for_plot = counts_for_plot[display_label_order]
    percentages_for_plot = percentages_for_plot[display_label_order]

    # --------------------------------------------------
    # Χι-τετράγωνο τεστ ανεξαρτησίας
    # --------------------------------------------------

    contingency_table = counts[VALID_LABELS]

    chi2_statistic, p_value, degrees_of_freedom, expected_counts = chi2_contingency(
        contingency_table
    )

    n = contingency_table.to_numpy().sum()

    min_dimension = min(
        contingency_table.shape[0] - 1,
        contingency_table.shape[1] - 1
    )

    if min_dimension > 0:
        cramers_v = math.sqrt(
            chi2_statistic / (n * min_dimension)
        )
    else:
        cramers_v = None

    chi_square_summary = pd.DataFrame({
        "metric": [
            "chi2_statistic",
            "p_value",
            "degrees_of_freedom",
            "total_records",
            "cramers_v"
        ],
        "value": [
            chi2_statistic,
            p_value,
            degrees_of_freedom,
            n,
            cramers_v
        ]
    })

    expected_counts_df = pd.DataFrame(
        expected_counts,
        index=contingency_table.index,
        columns=contingency_table.columns
    )

    # --------------------------------------------------
    # Αποθήκευση πινάκων
    # --------------------------------------------------

    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    counts.to_csv(
        counts_output_path,
        encoding="utf-8-sig"
    )

    percentages.to_csv(
        percentages_output_path,
        encoding="utf-8-sig"
    )

    chi_square_summary.to_csv(
        chi_square_output_path,
        index=False,
        encoding="utf-8-sig"
    )

    expected_counts_df.to_csv(
        expected_counts_output_path,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------
    # Γράφημα 1: Stacked bar chart με counts
    # --------------------------------------------------

    counts_for_plot.plot(
        kind="bar",
        stacked=True,
        figsize=(11, 6),
        color=display_color_order
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
    plt.savefig(
        stacked_bar_output_path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # --------------------------------------------------
    # Γράφημα 2: 100% stacked bar chart με ποσοστά
    # --------------------------------------------------

    percentages_for_plot.plot(
        kind="bar",
        stacked=True,
        figsize=(11, 6),
        color=display_color_order
    )

    plt.title("Ποσοστιαία Κατανομή Συναισθήματος ανά Θέμα")
    plt.xlabel("Θέμα")
    plt.ylabel("Ποσοστό Εγγραφών (%)")
    plt.xticks(rotation=30, ha="right")

    plt.legend(
        title="Συναίσθημα",
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )

    plt.tight_layout()
    plt.savefig(
        percent_stacked_bar_output_path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # --------------------------------------------------
    # Γράφημα 3: Heatmap topic × sentiment
    # --------------------------------------------------

    plt.figure(figsize=(9, 5))

    sns.heatmap(
        percentages_for_plot,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        cbar_kws={"label": "Ποσοστό εγγραφών (%)"}
    )

    plt.title("Θερμικός Χάρτης Συναισθήματος ανά Θέμα (%)")
    plt.xlabel("Συναίσθημα")
    plt.ylabel("Θέμα")
    plt.tight_layout()
    plt.savefig(
        heatmap_output_path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # --------------------------------------------------
    # Εκτύπωση αποτελεσμάτων
    # --------------------------------------------------

    print()
    print("Topic-sentiment counts:")
    print(counts.to_string())

    print()
    print("Topic-sentiment percentages:")
    print(percentages.to_string())

    print()
    print("Chi-square test:")
    print(chi_square_summary.to_string(index=False))

    print()
    print("Files saved:")
    print(counts_output_path)
    print(percentages_output_path)
    print(chi_square_output_path)
    print(expected_counts_output_path)
    print(stacked_bar_output_path)
    print(percent_stacked_bar_output_path)
    print(heatmap_output_path)


if __name__ == "__main__":
    main()