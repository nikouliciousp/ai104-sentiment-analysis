![Aegean College](./aegean.png)

# 💬 AI104 — Sentiment Analysis & Topic Classification

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![University of Essex](https://img.shields.io/badge/University%20of%20Essex-MSc%20AI-purple?style=for-the-badge)
![Module](https://img.shields.io/badge/Module-AI%20104%20ML%20%26%20Python-blue?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)

**MSc Artificial Intelligence** | University of Essex via Aegean College  
**Module:** AI 104 — Machine Learning with Python  
**Assignment:** Group Project  
**Team:** 5 members

---

## 📋 Description

A complete natural language processing pipeline applied to **1,056 technology-forum posts** collected from HackerNews across four topics. The project covers **manual annotation** with inter-annotator agreement, **custom feature scoring**, **topic classification** and **sentiment classification** using Naive Bayes, K-Nearest Neighbours and Random Forest — followed by temporal analysis, topic–sentiment association testing and informative keyword extraction.

---

## 🇬🇷 Σύνοψη

Ολοκληρωμένη ροή επεξεργασίας φυσικής γλώσσας σε 1.056 αναρτήσεις τεχνολογικού φόρουμ από το HackerNews, σε τέσσερα θέματα. Περιλαμβάνει **χειροκίνητο σχολιασμό** με μέτρηση συμφωνίας σχολιαστών (Fleiss' kappa), **προσαρμοσμένη βαθμολόγηση χαρακτηριστικών**, **κατηγοριοποίηση θεμάτων** και **ταξινόμηση συναισθήματος** με Naive Bayes, K-Κοντινότερους Γείτονες και Τυχαίο Δάσος, καθώς και χρονική ανάλυση, έλεγχο σχέσης θέματος–συναισθήματος και εξαγωγή πληροφοριακών λέξεων-κλειδιών.

---

## 👥 Team

| Member | Name | Role | Report Sections |
|:------:|:-----|:-----|:----------------|
| 1 | Kristiana Milonaki | Data & Coordination Lead | 1, 3, 4, 8.1, 8.2 |
| 2 | Vasilis Papadimitropoulos | Text & Features Lead | 5 |
| 3 | Agathoklis Krimperis | Topic Classification Lead | 6, 8.3 |
| 4 | Perikles Nikoulis | Sentiment & Analysis Lead | 7, 8.4 |
| 5 | Dimitrios Chatzis | Code Appendix & Reproducibility Lead | Appendix |

---

## 🏷️ Topics

| Topic | Posts |
|:------|:-----:|
| Artificial Intelligence | 253 |
| Cryptocurrency | 272 |
| Cybersecurity | 266 |
| Climate Change | 265 |

---

## 📊 Dataset

| Dataset | Source | Raw | Clean | Period | Description |
|:--------|:------:|:---:|:-----:|:------:|:------------|
| HackerNews posts | Algolia HN Search API | 1,211 | 1,056 | 2024–2026 | Technology-forum comments across four topics |

> ℹ️ Public API — **no credentials required**. Endpoint: https://hn.algolia.com/api

**Sentiment distribution (after majority voting):**

| Class | Count | Share |
|:------|:-----:|:-----:|
| Neutral | 620 | 58.7% |
| Negative | 357 | 33.8% |
| Positive | 79 | 7.5% |

---

## 📁 Repository Structure

| Folder | Content | Owner |
|:-------|:--------|:-----:|
| [`src/section_1_data_collection/`](src/section_1_data_collection/) | `hacker_news_collector.py` — Algolia API retrieval | Member 1 |
| [`src/section_2_annotation/`](src/section_2_annotation/) | annotation template, `majority_vote.py`, `resolve_manual_review.py` | Member 1 |
| [`src/section_3_text_cleaning/`](src/section_3_text_cleaning/) | `text_cleaning_hackernews.py` — deduplication, normalisation | Member 1 |
| [`src/section_4_feature_discovery/`](src/section_4_feature_discovery/) | term frequency, TF-IDF, n-grams, custom scoring | Member 2 |
| [`src/section_5_topic_classification/`](src/section_5_topic_classification/) | topic models, BoW vs TF-IDF, evaluation | Member 3 |
| [`src/section_6_sentiment_classification/`](src/section_6_sentiment_classification/) | `prepare_sentiment_dataset.py`, `sentiment_classification.py`, `sentiment_evaluation.py` | Member 4 |
| [`src/section_7_further_analysis/`](src/section_7_further_analysis/) | temporal, topic–sentiment, `informative_terms.py` | Members 1, 3, 4 |
| [`src/common/`](src/common/) | `synopsis.py`, `create_modeling_dataset.py` | Member 1 |

> ⚠️ `src/` folder numbering is internal and fixed early in the project. It does **not** map one-to-one to the report sections. In the report: Topic Discovery = **5**, Topic Classification = **6**, Sentiment Classification = **7**, Further Analysis = **8**.

---

## 📦 Required Python Packages

```python
packages = [
    "pandas",        # Data manipulation & analysis
    "numpy",         # Numerical computation
    "matplotlib",    # Plotting & visualization
    "seaborn",       # Statistical visualization
    "scikit-learn",  # Naive Bayes, KNN, Random Forest, TF-IDF, metrics
    "scipy",         # Chi-square test, paired t-test
    "requests",      # Algolia API calls
]
```

> ✅ All dependencies are listed in `requirements.txt` — install with `pip install -r requirements.txt`

---

## ▶️ How to Run

**1. Clone the repository**

```bash
git clone https://github.com/nikouliciousp/ai104-sentiment-analysis.git
cd ai104-sentiment-analysis
```

**2. Create virtual environment & install dependencies**

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
source .venv/bin/activate        # Linux/Mac

pip install -r requirements.txt
```

**3. Run the scripts in order** — each step depends on the previous

```bash
# Data preparation
python src/section_1_data_collection/hacker_news_collector.py
python src/section_3_text_cleaning/text_cleaning_hackernews.py
python src/section_2_annotation/create_hackernews_annotation_template.py
python src/section_2_annotation/majority_vote.py
python src/section_2_annotation/resolve_manual_review.py
python src/common/create_modeling_dataset.py

# Feature discovery & topic classification
python src/section_4_feature_discovery/topic_classification_features.py
python src/section_5_topic_classification/topic_classification.py

# Sentiment classification
python src/section_6_sentiment_classification/prepare_sentiment_dataset.py
python src/section_6_sentiment_classification/sentiment_classification.py
python src/section_6_sentiment_classification/sentiment_evaluation.py

# Further analysis
python src/section_7_further_analysis/temporal_sentiment_analysis.py
python src/section_7_further_analysis/topic_sentiment_analysis.py
python src/section_7_further_analysis/informative_terms.py
```

> ⚠️ **Important:** The annotation step requires four independent annotators. The completed annotation files are already committed to `data/annotated/`.

---

## 📈 Analysis Overview

### Annotation & Agreement

| Section | Topic | Key Method |
|:-------:|:------|:-----------|
| 4 | Annotation process | 4 annotators, majority voting, adjudication of ties |
| 4 | Inter-annotator agreement | Cohen's kappa (pairwise), Fleiss' kappa (overall) |

### Topic Discovery & Classification

| Section | Topic | Key Method |
|:-------:|:------|:-----------|
| 5 | Feature discovery | Term frequency, TF-IDF, bigrams, positional scoring |
| 5 | Custom scoring | Weighted composite score, leakage-safe fitting |
| 6 | Topic classification | Naive Bayes, KNN, Random Forest · BoW vs TF-IDF |
| 6 | Evaluation | Accuracy, precision, recall, macro-F1, confusion matrix |

### Sentiment Classification

| Section | Topic | Key Method |
|:-------:|:------|:-----------|
| 7 | Dataset preparation | Multiclass vs binary target, class weights |
| 7 | Model comparison | 3 models × 2 weighting configs × 2 targets |
| 7 | Evaluation | Confusion matrix, per-class metrics, best model selection |

### Further Analysis

| Section | Topic | Key Method |
|:-------:|:------|:-----------|
| 8.1 | Temporal analysis | Monthly & quarterly sentiment trends |
| 8.2 | Topic × sentiment | Chi-square test of independence, Cramér's V |
| 8.3 | Bigrams vs unigrams | Representation comparison for topic classification |
| 8.4 | Informative keywords | Naive Bayes log-probability discrimination score |

---

## 🔑 Key Findings

| Finding | Value |
|:--------|:------|
| Posts collected / retained | **1,211** raw → **1,056** clean |
| Class imbalance (sentiment) | **7.85 : 1** (neutral vs positive) |
| Inter-annotator agreement | Fleiss' kappa = **0.462** (moderate) |
| Majority-class baseline | 58.71% accuracy, but only **0.247** macro-F1 |
| Best topic model | **Random Forest + BoW**, macro-F1 = **0.986** |
| Best sentiment model | **Naive Bayes (weighted)**, macro-F1 = **0.433** |
| Effect of class weighting | Naive Bayes +0.118, Random Forest +0.084, KNN unchanged |
| Representation finding | **Naive Bayes prefers BoW** — TF-IDF violates the multinomial assumption |
| Hardest class | Positive (79 samples) — best F1 = 0.214 |
| Main confusion boundary | Neutral ↔ Negative — the same boundary annotators disagreed on |
| Sarcasm detected | `successful`, `sure` rank as **negative** discriminators |
| Reproducibility seed | `random_state=42`, stratified 80/20 split |

---

## 📂 Output Files

### Figures

| Directory | Content |
|:----------|:--------|
| `results/figures/section_4/` | Term frequency, TF-IDF and custom score charts |
| `results/figures/section_5/` | Topic model comparison, confusion matrix |
| `results/figures/section_6/` | Sentiment class distribution, model comparison, confusion matrix |
| `results/figures/section_7/` | Monthly/quarterly trends, topic × sentiment, informative keywords |

### Tables

| File | Content |
|:-----|:--------|
| `annotation_quality_summary.csv` | Annotation completeness and validity |
| `sentiment_distribution_summary.csv` | Class distribution before and after resolution |
| `sentiment_class_weights.csv` | Computed class weights for imbalance correction |
| `sentiment_target_options_comparison.csv` | Multiclass vs binary target evaluation |
| `sentiment_model_comparison.csv` | Full 3 × 2 × 2 experiment matrix |
| `sentiment_per_class_metrics.csv` | Precision, recall, F1 per sentiment class |
| `sentiment_representation_comparison.csv` | BoW vs TF-IDF per model |
| `sentiment_confusion_matrix.csv` | Confusion matrix of the best model |
| `topic_classification_bow_tfidf_comparison.csv` | Topic model comparison |
| `topic_sentiment_chi_square_test.csv` | Chi-square test of independence |
| `informative_unigrams_by_sentiment.csv` | Top discriminative words per sentiment |
| `informative_bigrams_by_sentiment.csv` | Top discriminative bigrams per sentiment |
| `frequent_vs_discriminative.csv` | Frequent terms contrasted with discriminative terms |

---

## 🔬 Reproducibility

| | |
|:---|:---|
| **Random seed** | `random_state = 42` across all experiments |
| **Split** | Stratified 80/20 (844 train / 212 test) |
| **Leakage prevention** | TF-IDF fitted on the training split only |
| **Feature validation** | Custom features verified invariant to the topic label |
| **Environment** | Python 3.10+, dependencies pinned in `requirements.txt` |

---

## 📚 References

- Alpaydın, E. (2020) *Introduction to Machine Learning*. 4th edn. Cambridge, MA: MIT Press
- Tan, P.-N., Steinbach, M., Karpatne, A. and Kumar, V. (2019) *Introduction to Data Mining*. 2nd edn. New York: Pearson
- Hand, D., Mannila, H. and Smyth, P. (2001) *Principles of Data Mining*. Cambridge, MA: MIT Press
- Cohen, J. (1960) A coefficient of agreement for nominal scales. *Educational and Psychological Measurement*, 20(1), 37–46
- Fleiss, J.L. (1971) Measuring nominal scale agreement among many raters. *Psychological Bulletin*, 76(5), 378–382
- Landis, J.R. and Koch, G.G. (1977) The measurement of observer agreement for categorical data. *Biometrics*, 33(1), 159–174
- Pedregosa, F. et al. (2011) Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830

---

## 📖 Citation

Y Combinator. (2024–2026). *HackerNews posts retrieved via Algolia HN Search API.*  
Retrieved July 2026, from https://hn.algolia.com/api

---

*All random operations use `random_state=42` for full reproducibility.*  
*Analysis conducted for academic purposes only.*
