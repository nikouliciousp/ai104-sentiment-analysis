![Aegean College](./aegean.png)

# 💬 AI104 — Sentiment Analysis & Topic Classification

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![University of Essex](https://img.shields.io/badge/University%20of%20Essex-MSc%20AI-purple?style=for-the-badge)
![Module](https://img.shields.io/badge/Module-AI%20104%20ML%20%26%20Python-blue?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)

**MSc Artificial Intelligence** | University of Essex via Aegean College  
**Module:** AI 104 — Machine Learning with Python  
**Lecturers:** PhD Candidate Dimitris Sakavalas, PhD Candidate Konstantinos Seretis, PhD Kyriakos Skoularikis  
**Assignment:** Group Assessment  
**Team:** 5 members

---

## 📋 Description

A complete natural language processing pipeline applied to **1,056 technology-forum posts** collected from Hacker News across four topics. The project covers **manual annotation** with inter-annotator agreement measurement, **feature discovery** (term frequency, TF-IDF, n-grams), **topic classification** using Naive Bayes, KNN, and Random Forest, and **sentiment analysis** with focus on class imbalance handling. The project additionally includes temporal sentiment analysis, topic–sentiment dependency analysis and discriminative keyword extraction.

---

## 🇬🇷 Σύνοψη

Ολοκληρωμένη ροή επεξεργασίας φυσικής γλώσσας σε 1.056 σύντομες αναρτήσεις τεχνολογικού περιεχομένου από το Hacker News, οργανωμένες σε τέσσερις θεματικές κατηγορίες. Το έργο καλύπτει τη χειροκίνητη επισήμανση συναισθήματος και τη μέτρηση της συμφωνίας μεταξύ σχολιαστών, την εξαγωγή χαρακτηριστικών μέσω συχνοτήτων όρων, TF-IDF και n-grams, την κατηγοριοποίηση θεμάτων με Naive Bayes, K-Nearest Neighbours και Random Forest και την κατηγοριοποίηση συναισθήματος με έμφαση στην αντιμετώπιση της ανισορροπίας των κλάσεων. Περιλαμβάνονται επίσης χρονική ανάλυση συναισθήματος, διερεύνηση της σχέσης θέματος–συναισθήματος και εξαγωγή πληροφοριακών λέξεων και διλέξων.

---

## 👥 Team

| Member | Name | Role | Report Sections |
|:------:|:-----|:-----|:----------------|
| 1 | Anna - Krystalia Mylonaki | Data & Coordination Lead | 1, 2, 3, 4, 8.1, 8.2, 8.3 |
| 2 | Vasilis Papadimitrakopoulos | Text & Features Lead | 5 |
| 3 | Agathoklis Krimpenis | Topic Classification Lead | 6 |
| 4 | Perikles Nikoulis | Sentiment & Analysis Lead | 7, 8.4 |
| 5 | Dimitrios Chatzikyriakidis | Code Appendix & Reproducibility Lead | 8.3, Appendix |

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
| Hacker News posts | Algolia HN Search API | 1,214 | 1,056 | 2023–2026 | Technology-forum comments across four topics |

> ℹ️ Public API — **no credentials required**. Endpoint: https://hn.algolia.com/api

**Final sentiment distribution after majority voting and adjudication:**

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
| [`src/section_7_further_analysis/`](src/section_7_further_analysis/) | temporal, topic–sentiment, `informative_terms.py` | Members 1, 4, 5 |
| [`src/common/`](src/common/) | `synopsis.py`, `create_modeling_dataset.py` | Member 1 |

> ⚠️ **Folder numbering clarification:**  
> The `src/` folder structure uses internal numbering (section_1 through section_7) that is **NOT** a direct one-to-one mapping to report sections.  
> **Report section mapping:**
> - Report Section 1 = Data Collection (src/section_1)
> - Report Section 2 = Text Cleaning (src/section_3)
> - Report Section 3 = Annotation & Agreement (src/section_2)
> - Report Section 4 = Inter-annotator Agreement Analysis (src/section_2)
> - Report Section 5 = Feature Discovery (src/section_4)
> - Report Section 6 = Topic Classification (src/section_5)
> - Report Section 7 = Sentiment Classification (src/section_6)
> - Report Section 8 = Further Analysis (src/section_7)

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
Jupyter is included as an optional environment for interactive inspection and partial execution of the code. The official reproducibility workflow is based on the Python scripts and the execution sequence documented in this README.

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

**3. Core execution sequence** — each step depends on the previous

To reproduce the reported results exactly, use the committed datasets and begin from the cleaning, feature-generation or modelling stages. Re-running the API collector may return different records because the source is continuously updated.

The following commands reproduce the core stages of the pipeline. Additional analysis scripts are documented in the corresponding `src/` folders.

```bash
# Optional: retrieve current data from the API.
# Skip this command when reproducing the reported results from committed data.

python src/section_1_data_collection/hacker_news_collector.py

# Data preparation (Report Sections 1–4)

python src/section_3_text_cleaning/text_cleaning_hackernews.py
python src/section_2_annotation/create_hackernews_annotation_template.py
python src/section_2_annotation/majority_vote.py
python src/section_2_annotation/resolve_manual_review.py
python src/common/create_modeling_dataset.py

# Feature discovery & topic classification (Report Sections 5–6)
python src/section_4_feature_discovery/topic_classification_features.py
python src/section_5_topic_classification/topic_classification.py

# Sentiment classification (Report Section 7)
python src/section_6_sentiment_classification/prepare_sentiment_dataset.py
python src/section_6_sentiment_classification/sentiment_classification.py
python src/section_6_sentiment_classification/sentiment_evaluation.py

# Further analysis (Report Section 8)
python src/section_7_further_analysis/temporal_sentiment_analysis.py
python src/section_7_further_analysis/topic_sentiment_analysis.py
python src/section_7_further_analysis/informative_terms.py
python src/section_7_further_analysis/sentiment_unigram_bigram_performance_comparison.py
python src/section_7_further_analysis/topic_classification_unigram_vs_bigram.py
```

> ⚠️ **Important:** The annotation step (section_2) requires four independent annotators. The completed annotation files are already committed to `data/annotated/`.

---

## 📈 Analysis Overview

### Annotation & Agreement (Report Sections 3–4)

| Section | Topic | Key Method | Owner |
|:-------:|:------|:-----------|:-----:|
| 3–4 | Annotation process | 4 annotators, majority voting, adjudication of ties | Member 1 |
| 3–4 | Inter-annotator agreement | Cohen's kappa (pairwise), Fleiss' kappa (overall) | Member 1 |

### Topic Discovery & Classification (Report Sections 5–6)

| Section | Topic | Key Method | Owner |
|:-------:|:------|:-----------|:-----:|
| 5 | Feature discovery | Term frequency, TF-IDF, bigrams, positional scoring | Member 2 |
| 5 | Custom scoring | Weighted composite score, leakage-safe fitting | Member 2 |
| 6 | Topic classification | Naive Bayes, KNN, Random Forest · BoW vs TF-IDF | Member 3 |
| 6 | Evaluation | Accuracy, precision, recall, macro-F1, confusion matrix | Member 3 |

### Sentiment Classification (Report Section 7)

| Section | Topic | Key Method | Owner |
|:-------:|:------|:-----------|:-----:|
| 7 | Dataset preparation | Multiclass vs binary target, class weights | Member 4 |
| 7 | Model comparison | 3 models × 2 weighting configs × 2 targets | Member 4 |
| 7 | Evaluation | Confusion matrix, per-class metrics, best model selection | Member 4 |

### Further Analysis (Report Section 8)

| Section | Topic | Key Method | Owner |
|:-------:|:------|:-----------|:-----:|
| 8.1 | Temporal analysis | Monthly & quarterly sentiment trends | Member 1 |
| 8.2 | Topic × sentiment | Chi-square test of independence, Cramér's V | Member 1 |
| 8.3 | Bigrams vs unigrams | Representation comparison for sentiment and topic classification | Members 1,5|
| 8.4 | Informative keywords | Naive Bayes log-probability discrimination score | Members 1, 4 |

---

## 🔑 Key Findings

| Finding | Value | Source |
|:--------|:------|:-------|
| Posts collected / retained | **1,214** raw → **1,056** clean | `annotation_quality_summary.csv` |
| Class imbalance (sentiment) | **7.85 : 1** (neutral vs positive) | `sentiment_distribution_summary.csv` |
| Inter-annotator agreement | Fleiss' kappa = **0.462** (moderate) | Report Section 4 |
| Majority-class baseline | **58.71%** accuracy, **0.247** macro-F1 | `sentiment_model_comparison.csv` |
| Best topic model | **Random Forest + BoW / TF-ID**, macro-F1 = **0.9861** | `topic_classification_bow_tfidf_comparison.csv` |
| Best sentiment model | **Naive Bayes (weighted)**, macro-F1 = **0.433** | `sentiment_model_comparison.csv` |
| Effect of class weighting | Naive Bayes +0.118, Random Forest +0.084, KNN unchanged | `sentiment_model_comparison.csv` |
| Representation finding | **Naive Bayes achieved higher macro-F1 with BoW** than with TF-IDF in this dataset; raw term counts are more closely aligned with the classical multinomial event model. | `sentiment_representation_comparison.csv` |
| Hardest class | Positive (79 samples) — best F1 = **0.214** | `sentiment_per_class_metrics.csv` |
| Main confusion boundary | Neutral ↔ Negative — same boundary where annotators disagreed | Report Section 7 |
| Potential contextual or ironic usage | `successful` and `sure` appear among negative discriminators and require contextual interpretation | `informative_unigrams_by_sentiment.csv` |
| **Random seed** | `random_state = 42` where supported by the relevant split or model | All scripts |

---

## 📂 Output Files

### Figures

| Directory | Content | Source Script |
|:----------|:--------|:-------------:|
| `results/figures/section_4/` | Term frequency, TF-IDF, custom score charts | `topic_classification_features.py` |
| `results/figures/section_5/` | Topic model comparison, confusion matrix | `topic_classification.py` |
| `results/figures/section_6/` | Sentiment class distribution, model comparison, confusion matrix | `sentiment_classification.py` |
| `results/figures/section_7/` | Monthly/quarterly trends, topic × sentiment, informative keywords | `temporal_sentiment_analysis.py`, `topic_sentiment_analysis.py`, `informative_terms.py` |
| `topic_unigram_bigram_macro_f1_comparison.png` | Macro-F1 comparison | `topic_classification_unigram_vs_bigram.py` |

### Tables

| File | Content | Generated By |
|:-----|:--------|:-------------:|
| `annotation_quality_summary.csv` | Annotation completeness and validity | `majority_vote.py` |
| `sentiment_distribution_summary.csv` | Class distribution before and after resolution | `prepare_sentiment_dataset.py` |
| `sentiment_class_weights.csv` | Computed class weights for imbalance correction | `sentiment_classification.py` |
| `sentiment_target_options_comparison.csv` | Multiclass vs binary target evaluation | `sentiment_evaluation.py` |
| `sentiment_model_comparison.csv` | Full 3 × 2 × 2 experiment matrix | `sentiment_evaluation.py` |
| `sentiment_per_class_metrics.csv` | Precision, recall, F1 per sentiment class | `sentiment_evaluation.py` |
| `sentiment_representation_comparison.csv` | BoW vs TF-IDF per model | `sentiment_evaluation.py` |
| `sentiment_confusion_matrix.csv` | Confusion matrix of the best model | `sentiment_evaluation.py` |
| `topic_classification_bow_tfidf_comparison.csv` | Topic model comparison | `topic_classification.py` |
| `topic_sentiment_chi_square_test.csv` | Chi-square test of independence | `topic_sentiment_analysis.py` |
| `informative_unigrams_by_sentiment.csv` | Top discriminative words per sentiment | `informative_terms.py` |
| `informative_bigrams_by_sentiment.csv` | Top discriminative bigrams per sentiment | `informative_terms.py` |
| `frequent_vs_discriminative.csv` | Frequent terms contrasted with discriminative terms | `informative_terms.py` |
| `topic_unigram_bigram_performance_comparison.csv` | Performance comparing | `topic_classification_unigram_vs_bigram.py` |
| `topic_unigram_bigram_class_metrics.csv` | Class metrics | `topic_classification_unigram_vs_bigram.py` |
| `topic_unigram_bigram_confusion_matrices.csv` | Confusion matrices | `topic_classification_unigram_vs_bigram.py` |
| `topic_frequent_bigrams_top30.csv` | Frequent bigrams top 30 | `topic_classification_unigram_vs_bigram.py` |
| `topic_frequent_bigrams_top30_filtered.csv` | Frequent bigrams top 30 filtered | `topic_classification_unigram_vs_bigram.py` |
| `topic_frequent_bigrams_by_class_top15.csv` | Frequent bigrams by class top 15 | `topic_classification_unigram_vs_bigram.py` |
| `topic_frequent_bigrams_by_class_top15_filtered.csv` | Frequent bigrams by class top 15 filtered | `topic_classification_unigram_vs_bigram.py` |

---

## 🔬 Reproducibility

| Aspect | Details |
|:-------|:--------|
| **Random seed** | `random_state = 42` across all experiments |
| **Split** | Stratified 80/20 (844 train / 212 test) |
| **Leakage prevention** | TF-IDF fitted on the training split only; custom features validated as topic-invariant |
| **Feature validation** | Custom features verified invariant to the topic label |
| **Environment** | Python 3.10+, dependencies listed in `requirements.txt` |
| **Annotation** | 4 independent annotators, plurality/majority voting, and manual adjudication of 2–2 ties by a fifth independent annotator |

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

Y Combinator. *Hacker News posts retrieved throug* the Algolia HN Search API** Dataset period: 06/07/2023–02/07/2026. Accessed July 2026. https://hn.algolia.com/api

---

*Where supported, stochastic procedures and data splits use `random_state=42` for full reproducibility.*  
*Analysis conducted for academic purposes only.*
