# Insurance-analytics-and-Predictive-Modeling-acis
Project Overview

This repository contains the implementation of Week 3’s challenge:
End-to-End Insurance Risk Analytics & Predictive Modeling for AlphaCare Insurance Solutions (ACIS).

The core business objective is to:

Analyze historical insurance claim data

Identify low-risk target groups for premium optimization

Support the marketing and underwriting teams with data-driven insights

Build reproducible and auditable analytics pipelines using DVC

Prepare for statistical hypothesis testing and predictive modeling

This repository is organized following professional data science and MLOps practices, including modular Python structure, DVC tracking, reproducible pipelines, and clear documentation.

🗂️ Repository Structure
Insurance-Analytics-and-Predictive-Modeling-ACIS/
│
├── data/                         # Raw and cleaned datasets (DVC-tracked)
│   ├── MachineLearningRating_v3.txt
│   ├── clean_data.csv
│   └── *.dvc                     # DVC pointer files
│
├── notebooks/
│   ├── task1_eda.ipynb           # Full EDA notebook
│   ├── task2_dvc_pipeline.ipynb  # DVC setup and pipeline notebook
│   └── EDA_preprocessing.ipynb   # Preprocessing exploration (if applicable)
│
├── src/
│   ├── data_loader.py            # Modular file loader
│   ├── cleaning.py               # Data cleaning pipeline
│   ├── preprocessing.py          # Preprocessing pipeline
│   └── run_cleaning.py           # Script executed through DVC pipeline
│
├── dvc_storage/                  # Local DVC remote storage
├── dvc.yaml                      # DVC pipeline definitions
├── dvc.lock                      # Locked pipeline versions
│
├── .gitignore
├── requirements.txt
└── README.md

🧠 Task 1: Exploratory Data Analysis (EDA)
🔍 Completed Work

Loaded dataset using modular src/data_loader.py

Cleaned dataset and standardized column names, data types, and missing values

Applied preprocessing steps:

Removed duplicates

Handled outliers via winsorization

Created engineered features:

lossratio = totalclaims / totalpremium

margin = totalpremium - totalclaims

risk_bucket (categorical segmentation)

Generated descriptive statistics for financial and customer attributes

Created 3+ advanced visualizations:

Province Risk Bubble Chart

Top Vehicle Models by Total Claims

Loss Ratio Distribution Density Plot

Conducted:

Univariate analysis

Bivariate analysis

Correlation heatmaps

Temporal trend analysis

Outlier analysis

📈 Key Insights

Strong geographic variation in loss ratios

Claims exhibit heavy right-tailed distribution

Vehicle model significantly affects risk patterns

Risk segmentation feasible using loss ratio buckets

Data contains seasonal claim patterns

Full results available in notebooks/task1_eda.ipynb.

📦 Task 2: Data Version Control (DVC)
✔ Achievements

Installed and initialized DVC (dvc init)

Created local remote storage (dvc_storage/)

Tracked raw and cleaned datasets (dvc add)

Implemented a DVC pipeline stage:

dvc stage add -n preprocess_clean_data \
   -d src/run_cleaning.py \
   -d data/MachineLearningRating_v3.txt \
   -o data/clean_data.csv \
   python src/run_cleaning.py


Successfully executed:

dvc repro
dvc push


Ensured full reproducibility of data preprocessing

DVC now manages:

Data lineage

Versioned datasets

Reproducible execution of cleaning steps

Experiment traceability
Setup Instructions
1. Clone repository
git clone https://github.com/<your-username>/Insurance-Analytics-and-Predictive-Modeling-ACIS.git
cd Insurance-Analytics-and-Predictive-Modeling-ACIS

2. Install dependencies
pip install -r requirements.txt

3. Reproduce pipeline
dvc repro

4. Push/pull data
dvc push
dvc pull

