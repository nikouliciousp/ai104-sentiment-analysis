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
├── raw/          → hackernews_posts_raw.csv (1200 posts, 4 topics)
├── clean/        → hackernews_posts_clean.csv
├── annotated/    → annotation_template.csv, hackernews_posts_annotated.csv
└── features/     → features_enriched.csv

notebooks/
├── section_4_feature_discovery.ipynb
├── section_5_topic_classification.ipynb
├── section_6_sentiment_classification.ipynb
└── section_7_further_analysis.ipynb

src/
├── hackernews_collector.py
├── text_cleaning.py
├── create_annotation_template.py
└── synopsis.py

results/
├── figures/
└── tables/

report/
└── omadiki_ergasia.docx

## Citation
Y Combinator. (2024–2026). HackerNews posts retrieved via Algolia HN Search API.
Retrieved July 2026, from https://hn.algolia.com/api