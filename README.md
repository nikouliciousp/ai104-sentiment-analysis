# AI104 — Sentiment Analysis & Topic Classification

MSc Artificial Intelligence | University of Essex / Aegean College
Group Assignment — AI 104: Machine Learning with Python

## Topics
- Artificial Intelligence
- Cryptocurrency
- Climate Change
- Remote Work

## Data Source
HackerNews via Algolia API (https://hn.algolia.com/api)
No credentials required — public API

## Team
- Member 1 — Data & Coordination Lead
- Member 2 — Text & Features Lead
- Member 3 — Classification Lead
- Member 4 — Sentiment & Analysis Lead

## Setup
pip install -r requirements.txt

## Structure

    data/
        raw/                                hackernews_posts_raw.csv
        clean/                              hackernews_posts_clean.csv
        annotated/                          annotation files (template, completed, final, resolved)
        modeling/                           hackernews_modeling_dataset.csv, sentiment_data.csv

    src/
        section_1_data_collection/          hacker_news_collector.py
        section_2_annotation/               create_hackernews_annotation_template.py,
                                            majority_vote.py, resolve_manual_review.py
        section_3_text_cleaning/            text_cleaning_hackernews.py
        section_4_feature_discovery/        (Member 2)
        section_5_topic_classification/     (Member 3)
        section_6_sentiment_classification/ prepare_sentiment_dataset.py,
                                            sentiment_classification.py
        section_7_further_analysis/         temporal_sentiment_analysis.py,
                                            topic_sentiment_analysis.py
        common/                             synopsis.py, create_modeling_dataset.py

    results/
        figures/   Section 6:               sentiment_class_distribution, sentiment_model_comparison
                   Section 7:               all monthly_*, quarterly_*, topic_* figures
        tables/    Section 2:               annotation_quality_summary, resolved_annotation_quality_summary,
                                            sentiment_distribution_summary, resolved_sentiment_distribution_summary
                   Common:                  modeling_dataset_quality_summary
                   Section 6:               sentiment_class_distribution, sentiment_class_weights,
                                            sentiment_target_options_comparison, sentiment_model_comparison,
                                            sentiment_per_class_metrics, sentiment_representation_comparison
                   Section 7:               monthly_*, quarterly_*, topic_*, topic_sentiment_chi_square_test

    report/

## Citation
Y Combinator. (2024-2026). HackerNews posts retrieved via Algolia HN Search API.
Retrieved July 2026, from https://hn.algolia.com/api