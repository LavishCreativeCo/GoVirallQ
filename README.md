# 🚀 GoViralIQ
### Predicting Instagram Engagement & Auditing Algorithmic Fairness Across Creator Niches

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-NLP-154f3c?style=flat-square)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=flat-square&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c?style=flat-square)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Notebooks-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)
![Status](https://img.shields.io/badge/Status-In%20Progress-purple?style=flat-square)

---

## 📌 Overview

**GoViralIQ** is an end-to-end machine learning and NLP pipeline that answers two questions the creator economy cares deeply about:

> **1. What makes Instagram content go viral?**
> **2. Does the algorithm predict engagement fairly across all creator niches?**

Using the [Viral Social Media Trends & Engagement Analysis](https://www.kaggle.com) dataset, this project builds predictive models for engagement rate, classifies content as viral or non-viral, performs sentiment analysis on post captions, and conducts a **fairness audit** — evaluating whether the model performs equally well across beauty, fitness, food, lifestyle, and fashion creators.

This mirrors the core systems used by **Meta's Feed Ranking team**, **Instagram's Explore algorithm**, and **Google's content recommendation infrastructure**.

---

## 🎯 Research Questions

- Can we accurately predict an Instagram post's engagement rate from its caption, hashtags, and metadata?
- Which features — sentiment, caption length, hashtag count, or posting time — drive virality most?
- Does the prediction model perform equally well across different creator niches?
- Are certain creator categories systematically over- or under-predicted?

---

## 🗂️ Project Structure

```
GoViralIQ/
│
├── 01_data_exploration.ipynb       # Initial EDA — load, inspect, engineer engagement_rate
├── 02_data_cleaning.ipynb          # Cleaning, outlier removal, normalization
├── 03_visualization.ipynb          # Matplotlib charts and full visual EDA
├── 04_ml_models.ipynb              # Regression + Classification models + evaluation
├── 05_nlp_pipeline.ipynb           # NLTK sentiment analysis + TF-IDF features
├── 06_fairness_audit.ipynb         # Fairness analysis across creator niches
│
├── data/
│   ├── raw/                        # Original Kaggle dataset
│   └── cleaned/                    # Processed and engineered dataset
│
├── requirements.txt                # All dependencies
└── README.md
```

---

## 🔬 Technical Pipeline

### Stage 1 — Data Ingestion & Wrangling `(01_data_exploration.ipynb)`
- Load Viral Social Media Trends dataset, filter for Instagram
- Explore shape, dtypes, null values, and distributions
- Engineer `engagement_rate = (likes + comments) / followers * 100`
- Create binary target: `viral = 1` if engagement_rate ≥ 80th percentile

### Stage 2 — Data Cleaning `(02_data_cleaning.ipynb)`
- Remove duplicates and handle missing caption data
- Outlier removal using IQR method on engagement metrics
- Normalize follower counts and engagement rates with MinMaxScaler
- Tokenize captions, remove stopwords, apply stemming (NLTK)

### Stage 3 — Visualization & EDA `(03_visualization.ipynb)`
- Engagement rate distributions by creator niche
- Correlation heatmap of all numerical features
- Word cloud of high-engagement caption keywords
- Sentiment score trends across posting patterns

### Stage 4 — Machine Learning Models `(04_ml_models.ipynb)`
| Model | Task | Metric |
|---|---|---|
| Linear Regression | Predict engagement rate | RMSE, R² |
| Random Forest Regressor | Predict engagement rate | RMSE, R² |
| Logistic Regression | Viral vs Non-Viral | Accuracy, F1 |
| Random Forest Classifier | Viral vs Non-Viral | Accuracy, F1 |

### Stage 5 — NLP Pipeline `(05_nlp_pipeline.ipynb)`
- Sentiment analysis on post captions using NLTK VADER
- Keyword extraction and TF-IDF vectorization
- Tokenization, stopword removal, stemming
- Sentiment score used as a feature in ML models

### Stage 6 — Fairness Audit ⚖️ `(06_fairness_audit.ipynb)`
- Split predictions by creator niche (beauty, fitness, food, lifestyle, fashion)
- Compare accuracy and error rates across groups
- Identify systematic over- or under-prediction by niche
- Visualize fairness gaps with grouped comparison charts
- Connect findings to creator monetization and algorithmic equity

---

## 📊 Dataset

**Source:** [Viral Social Media Trends & Engagement Analysis — Kaggle](https://www.kaggle.com)
**Author:** Atharva Soundankar | **Downloads:** 9,590+ | **Upvotes:** 105

| Feature | Description |
|---|---|
| `Platform` | Social media platform (filtered to Instagram) |
| `Likes` | Number of post likes |
| `Comments` | Number of post comments |
| `Followers` | Creator follower count |
| `Category` | Creator niche / content type |
| `engagement_rate` | Engineered: (likes + comments) / followers × 100 |
| `viral` | Engineered: 1 if engagement_rate ≥ 80th percentile |

> All data is publicly available on Kaggle. No private or client data is used.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data | Pandas, NumPy |
| Visualization | Matplotlib |
| Machine Learning | Scikit-Learn |
| NLP | NLTK, VADER Sentiment Analyzer |
| Environment | Google Colab, Jupyter Notebooks |
| Version Control | Git, GitHub |

---

## ▶️ How to Run

**Option 1 — Google Colab (Recommended)**
1. Open any `.ipynb` notebook file
2. Click **"Open in Colab"**
3. Upload your Kaggle CSV when prompted
4. Run all cells top to bottom

**Option 2 — Local**
```bash
git clone https://github.com/LavishCreativeCo/GoViralIQ.git
cd GoViralIQ
pip install -r requirements.txt
jupyter notebook
```

---

## 📓 Notebook Progress

| # | Notebook | Status |
|---|---|---|
| 01 | Data Exploration | ✅ Complete |
| 02 | Data Cleaning | 🔄 In Progress |
| 03 | Visualization | 🔜 Coming Soon |
| 04 | ML Models | 🔜 Coming Soon |
| 05 | NLP Pipeline | 🔜 Coming Soon |
| 06 | Fairness Audit | 🔜 Coming Soon |

---

## 🔗 Related Work

This project is part of a connected body of research on **algorithmic fairness**:

- 📄 [Fairness-Aware Deep Learning for Skin-Tone Classification](https://github.com/LavishCreativeCo) — Published Research, Mercy University Fall 2025
- 🎨 [MelaninMatch AI](https://github.com/LavishCreativeCo/MelaninMatchAI) — Shade-matching CNN with 92% accuracy across 40+ skin tones
- 🌟 [JuaShade](https://github.com/LavishCreativeCo/Jua-Shade) — Computer vision fairness pipeline for skin tone detection

---

## 👩🏽‍💻 Author

**Chastity Lewis**
Applied ML Engineer | Data Scientist | Full Stack AI Developer

[![Portfolio](https://img.shields.io/badge/Portfolio-Behance-1769FF?style=flat-square&logo=behance&logoColor=white)](https://behance.net/LavishCreativeco)
[![GitHub](https://img.shields.io/badge/GitHub-LavishCreativeCo-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LavishCreativeCo)
[![Email](https://img.shields.io/badge/Email-Clewis44@mercy.edu-EA4335?style=flat-square&logo=gmail&logoColor=white)](mailto:Clewis44@mercy.edu)

---

## 📚 Academic Context

Built as the final project for **CISC 540 — Computational Data Analysis**
Mercy University | Department of Mathematics and Computer Sciences | Spring 2026
Instructor: Dr. Tianyu Wang

---

*GoViralIQ is an independent academic project and is not affiliated with any commercial platform.*
