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
│   └── EDA_preprocessing.ipynb   # Preprocessing
    ├── Task3_hypothesis_testing1.ipynb
|   └── Task4_statiscial_modeling.ipynb
│
├── src/
│   ├── data_loader.py            # Modular file loader
│   ├── cleaning.py               # Data cleaning pipeline
│   ├── Effects.py          
|   ├── equivalence.py               
│   ├── Hypothesis_helper.py         
|   ├── Segementation.py               
│   ├── preprocessing.py          # Preprocessing pipeline
|   ├── statistical_tests.py               
│   ├── visualization.py         
│   └── run_cleaning.py           # Script executed through DVC pipeline
│
├── dvc_storage/                  # Local DVC remote storage
├── dvc.yaml                      # DVC pipeline definitions
│
├── .gitignore
├── requirements.txt
└── README.md

---

## Key Achievements

### 1. **Exploratory Data Analysis & Data Pipeline**
- Successfully loaded and cleaned 1M+ row pipe-delimited dataset
- Created critical business metrics:
  - `margin` = TotalPremium − TotalClaims
  - `has_claim` = binary claim indicator
  - `vehicle_age`, `power_ratio`, `province_risk`, `zipcode_risk`, `premium_to_sum_ratio`

### 2. **Statistical Hypothesis Testing** (Task 3)
Tested and **rejected** the following null hypotheses (α = 0.05):

| Hypothesis | Test Used | P-value | Result |
|----------|----------|--------|--------|
| No risk frequency difference across provinces | Chi-Squared | ~0.000000 | **Rejected** |
| No claim severity difference across provinces | One-Way ANOVA | 6.8e-6 | **Rejected** |
| (In progress) No risk difference between men/women | Chi-squared / t-test | — | TBD |
| (In progress) No margin difference between zip codes | ANOVA | — | TBD |

**Business Recommendation**: Implement **regional risk-based pricing** — increase premiums in Gauteng/Free State, offer discounts in Northern Cape.

### 3. **Predictive Modeling Pipeline Ready** (Task 4)
- Built scalable `ColumnTransformer` + `Pipeline` with:
  - Median imputation + scaling for numerical features
  - OneHotEncoding for categorical (with high-cardinality filtering)
- Feature engineered risk scores using target encoding (province/zipcode claim rates)
- Ready for XGBoost, RandomForest, and Linear Regression models
- `train_test_split` imported — next step: model training & SHAP interpretability

---

## How to Run the Project

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/KAIM-Week3-Insurance-Analytics.git
cd KAIM-Week3-Insurance-Analytics

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run notebooks in order
jupyter notebook
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

