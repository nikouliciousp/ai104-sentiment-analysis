"""
Topic modeling with LDA and NMF.

Fits unsupervised topic models on the full corpus (k=4 latent topics) and
compares discovered topics with the human-assigned topic labels used during
data collection. LDA uses a document-term count matrix; NMF uses TF-IDF.

Input : data/features/hackernews_topic_features_dataset.csv
Output: results/tables/section_4/topic_lda_top_terms.csv
        results/tables/section_4/topic_nmf_top_terms.csv
        results/tables/section_4/topic_lda_document_assignments.csv
        results/tables/section_4/topic_nmf_document_assignments.csv
        results/tables/section_4/topic_lda_label_crosstab.csv
        results/tables/section_4/topic_nmf_label_crosstab.csv
        results/tables/section_4/topic_modeling_alignment_summary.csv
        results/figures/section_4/topic_lda_top_terms.png
        results/figures/section_4/topic_nmf_top_terms.png
        results/figures/section_4/topic_modeling_label_heatmaps.png
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INPUT_PATH = os.path.join(
    BASE_DIR, "data", "features", "hackernews_topic_features_dataset.csv"
)

TABLES_DIR = os.path.join(BASE_DIR, "results", "tables", "section_4")
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures", "section_4")

LDA_TOP_TERMS_OUTPUT = os.path.join(TABLES_DIR, "topic_lda_top_terms.csv")
NMF_TOP_TERMS_OUTPUT = os.path.join(TABLES_DIR, "topic_nmf_top_terms.csv")
LDA_ASSIGNMENTS_OUTPUT = os.path.join(TABLES_DIR, "topic_lda_document_assignments.csv")
NMF_ASSIGNMENTS_OUTPUT = os.path.join(TABLES_DIR, "topic_nmf_document_assignments.csv")
LDA_CROSSTAB_OUTPUT = os.path.join(TABLES_DIR, "topic_lda_label_crosstab.csv")
NMF_CROSSTAB_OUTPUT = os.path.join(TABLES_DIR, "topic_nmf_label_crosstab.csv")
ALIGNMENT_OUTPUT = os.path.join(TABLES_DIR, "topic_modeling_alignment_summary.csv")
LDA_FIGURE_OUTPUT = os.path.join(FIGURES_DIR, "topic_lda_top_terms.png")
NMF_FIGURE_OUTPUT = os.path.join(FIGURES_DIR, "topic_nmf_top_terms.png")
HEATMAP_FIGURE_OUTPUT = os.path.join(FIGURES_DIR, "topic_modeling_label_heatmaps.png")

TEXT_COLUMN = "text_no_stopwords"
N_TOPICS = 4
TOP_TERMS = 15
RANDOM_STATE = 42

VECTORIZER_PARAMS = {
    "lowercase": True,
    "ngram_range": (1, 1),
    "min_df": 2,
    "max_df": 0.9,
}

LATENT_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def extract_top_terms(model, feature_names, n_terms):
    """Return top weighted terms for each latent topic."""
    rows = []

    for topic_id, topic_weights in enumerate(model.components_):
        top_indices = np.argsort(topic_weights)[::-1][:n_terms]

        for rank, feature_index in enumerate(top_indices, start=1):
            rows.append({
                "latent_topic": topic_id,
                "rank": rank,
                "term": feature_names[feature_index],
                "weight": round(float(topic_weights[feature_index]), 6),
            })

    return pd.DataFrame(rows)


def build_document_assignments(df, topic_matrix, model_name):
    """Attach dominant latent topic and per-topic probabilities to each document."""
    dominant_topics = topic_matrix.argmax(axis=1)
    max_probs = topic_matrix.max(axis=1)

    output_df = df[["item_id", "post_id", "topic", TEXT_COLUMN]].copy()
    output_df = output_df.rename(columns={"topic": "actual_topic"})
    output_df["model"] = model_name
    output_df["dominant_latent_topic"] = dominant_topics
    output_df["dominant_topic_probability"] = np.round(max_probs, 6)

    for topic_id in range(topic_matrix.shape[1]):
        output_df[f"latent_topic_{topic_id}_prob"] = np.round(
            topic_matrix[:, topic_id], 6
        )

    return output_df


def build_label_crosstab(actual_topics, latent_topics):
    """Count matrix of actual topic labels vs dominant latent topics."""
    crosstab = pd.crosstab(
        actual_topics,
        latent_topics,
        rownames=["actual_topic"],
        colnames=["latent_topic"],
        dropna=False,
    )

    return crosstab


def build_row_normalized_crosstab(crosstab):
    """Normalize each actual-topic row to percentages for heatmap display."""
    row_totals = crosstab.sum(axis=1)
    normalized = crosstab.div(row_totals, axis=0) * 100
    return normalized.round(2)


def compute_alignment_accuracy(actual_topics, latent_topics):
    """
    Find the best one-to-one mapping between latent and actual topics and
    return alignment accuracy (Hungarian algorithm on the crosstab counts).
    """
    crosstab = build_label_crosstab(actual_topics, latent_topics)
    cost_matrix = crosstab.values

    row_ind, col_ind = linear_sum_assignment(-cost_matrix)

    matched = 0
    for actual_index, latent_index in zip(row_ind, col_ind):
        matched += cost_matrix[actual_index, latent_index]

    accuracy = matched / len(actual_topics)
    return round(accuracy, 4), crosstab


def plot_top_terms(top_terms_df, output_path, title):
    """Horizontal bar charts of top terms per latent topic."""
    fig, axes = plt.subplots(
        nrows=N_TOPICS,
        ncols=1,
        figsize=(10, 3.2 * N_TOPICS),
        squeeze=False,
    )

    for latent_topic, ax in enumerate(axes.flatten()):
        topic_data = (
            top_terms_df[top_terms_df["latent_topic"] == latent_topic]
            .sort_values("weight", ascending=True)
            .tail(TOP_TERMS)
        )

        color = LATENT_COLORS[latent_topic % len(LATENT_COLORS)]

        ax.barh(
            topic_data["term"],
            topic_data["weight"],
            color=color,
            alpha=0.85,
        )
        ax.set_title(f"Latent topic {latent_topic}")
        ax.set_xlabel("Component weight")
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(title, y=1.01, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_label_heatmaps(lda_crosstab_pct, nmf_crosstab_pct, output_path):
    """Side-by-side heatmaps of actual vs latent topic alignment."""
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))

    for ax, crosstab_pct, title in zip(
        axes,
        [lda_crosstab_pct, nmf_crosstab_pct],
        ["LDA — actual vs latent topic (%)", "NMF — actual vs latent topic (%)"],
    ):
        sns.heatmap(
            crosstab_pct,
            annot=True,
            fmt=".1f",
            cmap="Blues",
            linewidths=0.5,
            cbar_kws={"label": "% of actual-topic documents"},
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Dominant latent topic")
        ax.set_ylabel("Actual topic label")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fit_lda(texts):
    """Fit LDA on a count matrix."""
    vectorizer = CountVectorizer(**VECTORIZER_PARAMS)
    matrix = vectorizer.fit_transform(texts)

    model = LatentDirichletAllocation(
        n_components=N_TOPICS,
        random_state=RANDOM_STATE,
        max_iter=30,
        learning_method="batch",
    )
    topic_matrix = model.fit_transform(matrix)

    return model, vectorizer, matrix, topic_matrix


def fit_nmf(texts):
    """Fit NMF on a TF-IDF matrix."""
    vectorizer = TfidfVectorizer(**VECTORIZER_PARAMS)
    matrix = vectorizer.fit_transform(texts)

    model = NMF(
        n_components=N_TOPICS,
        random_state=RANDOM_STATE,
        max_iter=300,
    )
    topic_matrix = model.fit_transform(matrix)

    return model, vectorizer, matrix, topic_matrix


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Topic features dataset not found: {INPUT_PATH}\n"
            "Run topic_text_preparation.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = [TEXT_COLUMN, "topic", "item_id", "post_id"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=[TEXT_COLUMN, "topic"]).reset_index(drop=True)
    df = df[df[TEXT_COLUMN].str.strip() != ""].reset_index(drop=True)

    texts = df[TEXT_COLUMN].tolist()
    actual_topics = df["topic"].tolist()

    print(f"Loaded {len(df)} records")
    print(f"Latent topics (k): {N_TOPICS}")
    print(f"Vectoriser params: {VECTORIZER_PARAMS}")
    print()

    lda_model, lda_vectorizer, lda_matrix, lda_topic_matrix = fit_lda(texts)
    nmf_model, nmf_vectorizer, nmf_matrix, nmf_topic_matrix = fit_nmf(texts)

    lda_top_terms = extract_top_terms(
        lda_model, lda_vectorizer.get_feature_names_out(), TOP_TERMS
    )
    nmf_top_terms = extract_top_terms(
        nmf_model, nmf_vectorizer.get_feature_names_out(), TOP_TERMS
    )

    lda_assignments = build_document_assignments(df, lda_topic_matrix, "lda")
    nmf_assignments = build_document_assignments(df, nmf_topic_matrix, "nmf")

    lda_dominant = lda_topic_matrix.argmax(axis=1)
    nmf_dominant = nmf_topic_matrix.argmax(axis=1)

    lda_accuracy, lda_crosstab = compute_alignment_accuracy(actual_topics, lda_dominant)
    nmf_accuracy, nmf_crosstab = compute_alignment_accuracy(actual_topics, nmf_dominant)

    lda_crosstab_pct = build_row_normalized_crosstab(lda_crosstab)
    nmf_crosstab_pct = build_row_normalized_crosstab(nmf_crosstab)

    alignment_summary = pd.DataFrame([
        {
            "model": "lda",
            "n_topics": N_TOPICS,
            "vocabulary_size": len(lda_vectorizer.get_feature_names_out()),
            "matrix_shape_rows": lda_matrix.shape[0],
            "matrix_shape_features": lda_matrix.shape[1],
            "alignment_accuracy": lda_accuracy,
        },
        {
            "model": "nmf",
            "n_topics": N_TOPICS,
            "vocabulary_size": len(nmf_vectorizer.get_feature_names_out()),
            "matrix_shape_rows": nmf_matrix.shape[0],
            "matrix_shape_features": nmf_matrix.shape[1],
            "alignment_accuracy": nmf_accuracy,
        },
    ])

    lda_top_terms.to_csv(LDA_TOP_TERMS_OUTPUT, index=False)
    nmf_top_terms.to_csv(NMF_TOP_TERMS_OUTPUT, index=False)
    lda_assignments.to_csv(LDA_ASSIGNMENTS_OUTPUT, index=False)
    nmf_assignments.to_csv(NMF_ASSIGNMENTS_OUTPUT, index=False)
    lda_crosstab.to_csv(LDA_CROSSTAB_OUTPUT)
    nmf_crosstab.to_csv(NMF_CROSSTAB_OUTPUT)
    lda_crosstab_pct.to_csv(
        os.path.join(TABLES_DIR, "topic_lda_label_crosstab_pct.csv")
    )
    nmf_crosstab_pct.to_csv(
        os.path.join(TABLES_DIR, "topic_nmf_label_crosstab_pct.csv")
    )
    alignment_summary.to_csv(ALIGNMENT_OUTPUT, index=False)

    plot_top_terms(lda_top_terms, LDA_FIGURE_OUTPUT, "LDA — top terms per latent topic")
    plot_top_terms(nmf_top_terms, NMF_FIGURE_OUTPUT, "NMF — top terms per latent topic")
    plot_label_heatmaps(lda_crosstab_pct, nmf_crosstab_pct, HEATMAP_FIGURE_OUTPUT)

    print("Alignment with actual topic labels (best one-to-one mapping):")
    print(f"  LDA accuracy: {lda_accuracy:.4f}")
    print(f"  NMF accuracy: {nmf_accuracy:.4f}")
    print()
    print("LDA top 8 terms per latent topic:")
    print()
    for latent_topic in range(N_TOPICS):
        print(f"--- Latent topic {latent_topic} ---")
        topic_terms = lda_top_terms[lda_top_terms["latent_topic"] == latent_topic].head(8)
        for _, row in topic_terms.iterrows():
            print(f"  {row['term']:20s} weight={row['weight']:.4f}")
        print()

    print("NMF top 8 terms per latent topic:")
    print()
    for latent_topic in range(N_TOPICS):
        print(f"--- Latent topic {latent_topic} ---")
        topic_terms = nmf_top_terms[nmf_top_terms["latent_topic"] == latent_topic].head(8)
        for _, row in topic_terms.iterrows():
            print(f"  {row['term']:20s} weight={row['weight']:.4f}")
        print()

    print("Saved:")
    print(LDA_TOP_TERMS_OUTPUT)
    print(NMF_TOP_TERMS_OUTPUT)
    print(LDA_ASSIGNMENTS_OUTPUT)
    print(NMF_ASSIGNMENTS_OUTPUT)
    print(LDA_CROSSTAB_OUTPUT)
    print(NMF_CROSSTAB_OUTPUT)
    print(ALIGNMENT_OUTPUT)
    print(LDA_FIGURE_OUTPUT)
    print(NMF_FIGURE_OUTPUT)
    print(HEATMAP_FIGURE_OUTPUT)


if __name__ == "__main__":
    main()
