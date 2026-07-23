import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

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

# --------------------------------------------------
# Πίνακες εξόδου
# --------------------------------------------------

monthly_counts_output_path = os.path.join(
    tables_dir,
    "monthly_sentiment_counts.csv"
)

monthly_percentages_output_path = os.path.join(
    tables_dir,
    "monthly_sentiment_percentages.csv"
)

quarterly_counts_output_path = os.path.join(
    tables_dir,
    "quarterly_sentiment_counts.csv"
)

quarterly_percentages_output_path = os.path.join(
    tables_dir,
    "quarterly_sentiment_percentages.csv"
)

topic_quarterly_counts_output_path = os.path.join(
    tables_dir,
    "topic_quarterly_sentiment_counts.csv"
)

topic_quarterly_percentages_output_path = os.path.join(
    tables_dir,
    "topic_quarterly_sentiment_percentages.csv"
)

# --------------------------------------------------
# Γραφήματα εξόδου
# --------------------------------------------------

quarterly_overall_trend_output_path = os.path.join(
    figures_dir,
    "quarterly_sentiment_trend_overall.png"
)

quarterly_stacked_bar_output_path = os.path.join(
    figures_dir,
    "quarterly_sentiment_100_percent_stacked_bar.png"
)

quarterly_negative_topic_trend_output_path = os.path.join(
    figures_dir,
    "quarterly_negative_sentiment_by_topic.png"
)

quarterly_negative_heatmap_output_path = os.path.join(
    figures_dir,
    "quarterly_negative_sentiment_heatmap_by_topic.png"
)

quarterly_small_multiples_output_path = os.path.join(
    figures_dir,
    "quarterly_sentiment_small_multiples_by_topic.png"
)

quarterly_sentiment_heatmaps_output_path = os.path.join(
    figures_dir,
    "quarterly_sentiment_heatmaps_by_sentiment.png"
)

# --------------------------------------------------
# Ρυθμίσεις εμφάνισης
# --------------------------------------------------

VALID_LABELS = ["negative", "neutral", "positive"]

SENTIMENT_COLORS = {
    "negative": "#D9534F",  # απαλό κόκκινο
    "neutral": "#AAB7C4",   # γκρι-μπλε
    "positive": "#6BBF59"   # πιο ήπιο πράσινο
}

SENTIMENT_LABELS_GR = {
    "negative": "Αρνητικό",
    "neutral": "Ουδέτερο",
    "positive": "Θετικό"
}


# --------------------------------------------------
# Βοηθητικές συναρτήσεις
# --------------------------------------------------

def format_quarter(period):
    """
    Μετατρέπει ένα pandas Period τύπου 2024Q1 σε πιο ευανάγνωστη μορφή,
    π.χ. "Ιαν–Μαρ 2024".
    """

    quarter_months = {
        1: "Ιαν–Μαρ",
        2: "Απρ–Ιουν",
        3: "Ιουλ–Σεπ",
        4: "Οκτ–Δεκ"
    }

    return f"{quarter_months[period.quarter]} {period.year}"


def rename_sentiment_columns_for_display(df):
    """
    Μετονομάζει τις στήλες sentiment μόνο για εμφάνιση στα γραφήματα.
    Δεν αλλάζει τα πραγματικά labels στο dataset.
    """

    return df.rename(columns=SENTIMENT_LABELS_GR)


# --------------------------------------------------
# Κύρια ανάλυση
# --------------------------------------------------

def main():
    df = pd.read_csv(input_path)

    print("Loaded resolved final dataset:", len(df), "records")

    required_columns = ["created_at", "topic", "final_sentiment"]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df[df["final_sentiment"].isin(VALID_LABELS)].copy()

    print("Valid sentiment records:", len(df))

    # --------------------------------------------------
    # Μετατροπή ημερομηνίας
    # --------------------------------------------------

    df["created_at"] = pd.to_datetime(
        df["created_at"],
        errors="coerce",
        utc=True
    )

    missing_dates = df["created_at"].isna().sum()

    if missing_dates > 0:
        print("Warning: rows with invalid dates:", missing_dates)

    df = df.dropna(subset=["created_at"]).copy()

    # Αφαιρούμε το timezone για period aggregation.
    df["created_at_no_tz"] = df["created_at"].dt.tz_convert(None)

    # --------------------------------------------------
    # Δημιουργία μηνιαίας και τριμηνιαίας μεταβλητής
    # --------------------------------------------------

    df["month"] = df["created_at_no_tz"].dt.to_period("M").astype(str)

    df["quarter_period"] = df["created_at_no_tz"].dt.to_period("Q")
    df["quarter"] = df["quarter_period"].apply(format_quarter)

    quarter_order = [
        format_quarter(period)
        for period in sorted(df["quarter_period"].unique())
    ]

    df["quarter"] = pd.Categorical(
        df["quarter"],
        categories=quarter_order,
        ordered=True
    )

    print()
    print("Temporal coverage:")
    print("Start:", df["created_at"].min())
    print("End:", df["created_at"].max())
    print("Unique months:", df["month"].nunique())
    print("Unique quarters:", df["quarter"].nunique())

    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # --------------------------------------------------
    # Μηνιαία κατανομή sentiment
    # --------------------------------------------------

    monthly_counts = pd.crosstab(
        df["month"],
        df["final_sentiment"]
    )

    for label in VALID_LABELS:
        if label not in monthly_counts.columns:
            monthly_counts[label] = 0

    monthly_counts = monthly_counts[VALID_LABELS]
    monthly_counts["total"] = monthly_counts.sum(axis=1)

    monthly_percentages = monthly_counts[VALID_LABELS].div(
        monthly_counts["total"],
        axis=0
    ) * 100

    monthly_percentages = monthly_percentages.round(1)
    monthly_percentages["total"] = monthly_counts["total"]

    monthly_counts.to_csv(
        monthly_counts_output_path,
        encoding="utf-8-sig"
    )

    monthly_percentages.to_csv(
        monthly_percentages_output_path,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------
    # Τριμηνιαία συνολική κατανομή sentiment
    # --------------------------------------------------

    quarterly_counts = pd.crosstab(
        df["quarter"],
        df["final_sentiment"]
    )

    quarterly_counts = quarterly_counts.sort_index()

    for label in VALID_LABELS:
        if label not in quarterly_counts.columns:
            quarterly_counts[label] = 0

    quarterly_counts = quarterly_counts[VALID_LABELS]
    quarterly_counts["total"] = quarterly_counts.sum(axis=1)

    quarterly_percentages = quarterly_counts[VALID_LABELS].div(
        quarterly_counts["total"],
        axis=0
    ) * 100

    quarterly_percentages = quarterly_percentages.round(1)
    quarterly_percentages["total"] = quarterly_counts["total"]

    quarterly_counts.to_csv(
        quarterly_counts_output_path,
        encoding="utf-8-sig"
    )

    quarterly_percentages.to_csv(
        quarterly_percentages_output_path,
        encoding="utf-8-sig"
    )

    # Display dataframe για ελληνικά legends.
    quarterly_percentages_display = rename_sentiment_columns_for_display(
        quarterly_percentages[VALID_LABELS]
    )

    display_label_order = [
        SENTIMENT_LABELS_GR[label]
        for label in VALID_LABELS
    ]

    display_color_order = [
        SENTIMENT_COLORS[label]
        for label in VALID_LABELS
    ]

    # --------------------------------------------------
    # Τριμηνιαία κατανομή sentiment ανά topic
    # --------------------------------------------------

    topic_quarterly_counts = (
        df.groupby(
            ["topic", "quarter", "final_sentiment"],
            observed=True
        )
        .size()
        .reset_index(name="count")
    )

    topic_quarterly_counts.to_csv(
        topic_quarterly_counts_output_path,
        index=False,
        encoding="utf-8-sig"
    )

    topic_quarterly_totals = (
        topic_quarterly_counts
        .groupby(
            ["topic", "quarter"],
            observed=True
        )["count"]
        .transform("sum")
    )

    topic_quarterly_percentages = topic_quarterly_counts.copy()

    topic_quarterly_percentages["percentage"] = (
        topic_quarterly_percentages["count"] / topic_quarterly_totals * 100
    ).round(1)

    topic_quarterly_percentages.to_csv(
        topic_quarterly_percentages_output_path,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------
    # Γράφημα 1: Τριμηνιαία συνολική εξέλιξη sentiment
    # --------------------------------------------------

    quarterly_percentages_display.plot(
        figsize=(11, 6),
        marker="o",
        color=display_color_order
    )

    plt.title("Τριμηνιαία Εξέλιξη Συναισθήματος")
    plt.xlabel("Τρίμηνο")
    plt.ylabel("Ποσοστό Εγγραφών (%)")
    plt.xticks(rotation=45, ha="right")

    plt.legend(
        title="Συναίσθημα",
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )

    plt.tight_layout()
    plt.savefig(
        quarterly_overall_trend_output_path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # --------------------------------------------------
    # Γράφημα 2: Τριμηνιαία 100% stacked κατανομή
    # --------------------------------------------------

    quarterly_percentages_display.plot(
        kind="bar",
        stacked=True,
        figsize=(11, 6),
        color=display_color_order
    )

    plt.title("Τριμηνιαία Ποσοστιαία Κατανομή Συναισθήματος")
    plt.xlabel("Τρίμηνο")
    plt.ylabel("Ποσοστό Εγγραφών (%)")
    plt.xticks(rotation=45, ha="right")

    plt.legend(
        title="Συναίσθημα",
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )

    plt.tight_layout()
    plt.savefig(
        quarterly_stacked_bar_output_path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # --------------------------------------------------
    # Γράφημα 3: Τριμηνιαία εξέλιξη negative sentiment ανά topic
    # --------------------------------------------------

    negative_df = topic_quarterly_percentages[
        topic_quarterly_percentages["final_sentiment"] == "negative"
    ].copy()

    negative_pivot = negative_df.pivot(
        index="quarter",
        columns="topic",
        values="percentage"
    )

    negative_pivot = negative_pivot.reindex(quarter_order)

    negative_pivot.columns = [
        str(col).replace("_", " ")
        for col in negative_pivot.columns
    ]

    negative_pivot.plot(
        figsize=(11, 6),
        marker="o"
    )

    plt.title("Τριμηνιαία Εξέλιξη Αρνητικού Συναισθήματος ανά Θέμα")
    plt.xlabel("Τρίμηνο")
    plt.ylabel("Ποσοστό Αρνητικών Εγγραφών (%)")
    plt.xticks(rotation=45, ha="right")

    plt.legend(
        title="Θέμα",
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )

    plt.tight_layout()
    plt.savefig(
        quarterly_negative_topic_trend_output_path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # --------------------------------------------------
    # Γράφημα 4: Heatmap negative sentiment ανά topic
    # --------------------------------------------------

    plt.figure(figsize=(12, 5))

    sns.heatmap(
        negative_pivot.T,
        annot=True,
        fmt=".1f",
        cmap="Reds",
        mask=negative_pivot.T.isna(),
        cbar_kws={"label": "Ποσοστό Αρνητικών Εγγραφών (%)"}
    )

    plt.title("Θερμικός Χάρτης Τριμηνιαίου Αρνητικού Συναισθήματος ανά Θέμα")
    plt.xlabel("Τρίμηνο")
    plt.ylabel("Θέμα")
    plt.tight_layout()
    plt.savefig(
        quarterly_negative_heatmap_output_path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # --------------------------------------------------
    # Γράφημα 5: Small multiples 100% stacked bars ανά topic
    # --------------------------------------------------

    topic_list = sorted(df["topic"].unique())

    fig, axes = plt.subplots(
        nrows=len(topic_list),
        ncols=1,
        figsize=(15, 4.2 * len(topic_list)),
        sharex=True
    )

    if len(topic_list) == 1:
        axes = [axes]

    for ax, topic in zip(axes, topic_list):
        topic_df = topic_quarterly_percentages[
            topic_quarterly_percentages["topic"] == topic
        ].copy()

        topic_pivot = topic_df.pivot(
            index="quarter",
            columns="final_sentiment",
            values="percentage"
        )

        topic_pivot = topic_pivot.reindex(quarter_order)

        for label in VALID_LABELS:
            if label not in topic_pivot.columns:
                topic_pivot[label] = 0

        topic_pivot = topic_pivot[VALID_LABELS]
        topic_pivot_for_plot = topic_pivot.fillna(0)
        topic_pivot_for_plot = rename_sentiment_columns_for_display(
            topic_pivot_for_plot
        )

        topic_pivot_for_plot = topic_pivot_for_plot[display_label_order]

        topic_pivot_for_plot.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            width=0.85,
            legend=False,
            color=display_color_order
        )

        ax.set_title(
            str(topic).replace("_", " "),
            fontsize=12,
            fontweight="bold",
            pad=10
        )

        ax.set_ylabel("Ποσοστό (%)")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.25)

        if ax != axes[-1]:
            ax.tick_params(labelbottom=False)

    legend_handles = [
        Patch(
            facecolor=SENTIMENT_COLORS[label],
            label=SENTIMENT_LABELS_GR[label]
        )
        for label in VALID_LABELS
    ]

    fig.legend(
        handles=legend_handles,
        title="Συναίσθημα",
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.02)
    )

    axes[-1].set_xlabel("Τρίμηνο")
    axes[-1].tick_params(axis="x", rotation=45)

    fig.subplots_adjust(
        top=0.96,
        bottom=0.12,
        hspace=0.45
    )

    plt.savefig(
        quarterly_small_multiples_output_path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # --------------------------------------------------
    # Γράφημα 6: Heatmaps ανά sentiment
    # --------------------------------------------------

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(12, 12)
    )

    sentiment_titles = {
        "negative": "Αρνητικό Συναίσθημα",
        "neutral": "Ουδέτερο Συναίσθημα",
        "positive": "Θετικό Συναίσθημα"
    }

    sentiment_cmaps = {
        "negative": "Reds",
        "neutral": "Greys",
        "positive": "Greens"
    }

    for ax, sentiment in zip(axes, VALID_LABELS):
        sentiment_df = topic_quarterly_percentages[
            topic_quarterly_percentages["final_sentiment"] == sentiment
        ].copy()

        sentiment_pivot = sentiment_df.pivot(
            index="quarter",
            columns="topic",
            values="percentage"
        )

        sentiment_pivot = sentiment_pivot.reindex(quarter_order)

        sentiment_pivot.columns = [
            str(col).replace("_", " ")
            for col in sentiment_pivot.columns
        ]

        sns.heatmap(
            sentiment_pivot.T,
            annot=True,
            fmt=".1f",
            cmap=sentiment_cmaps[sentiment],
            mask=sentiment_pivot.T.isna(),
            cbar_kws={"label": "Ποσοστό (%)"},
            ax=ax
        )

        ax.set_title(sentiment_titles[sentiment])
        ax.set_xlabel("Τρίμηνο")
        ax.set_ylabel("Θέμα")

    plt.tight_layout()
    plt.savefig(
        quarterly_sentiment_heatmaps_output_path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # --------------------------------------------------
    # Εκτύπωση σύνοψης στο terminal
    # --------------------------------------------------

    print()
    print("Quarterly sentiment counts:")
    print(quarterly_counts.to_string())

    print()
    print("Quarterly sentiment percentages:")
    print(quarterly_percentages.to_string())

    print()
    print("Files saved:")
    print(monthly_counts_output_path)
    print(monthly_percentages_output_path)
    print(quarterly_counts_output_path)
    print(quarterly_percentages_output_path)
    print(topic_quarterly_counts_output_path)
    print(topic_quarterly_percentages_output_path)
    print(quarterly_overall_trend_output_path)
    print(quarterly_stacked_bar_output_path)
    print(quarterly_negative_topic_trend_output_path)
    print(quarterly_negative_heatmap_output_path)
    print(quarterly_small_multiples_output_path)
    print(quarterly_sentiment_heatmaps_output_path)


if __name__ == "__main__":
    main()