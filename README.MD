# AlphaCare Insurance Solutions – Risk Analytics & Predictive Pricing  
**End-to-End Insurance Analytics and Predictive Modeling**  
**10 Academy – KAIM Week 3 Challenge** | December 2025

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-blue)
![Scikit--Learn](https://img.shields.io/badge/Scikit_Learn-1.3+-orange)
![Seaborn](https://img.shields.io/badge/Seaborn-Latest-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Project Overview

This repository contains a complete **insurance risk analytics and predictive modeling pipeline** built for **AlphaCare Insurance Solutions (ACIS)** using real South African car insurance data (Feb 2014 – Aug 2015, ~1M policy records).

**Business Goal**:  
Identify low-risk customer segments to enable targeted premium reductions, improve profitability, and gain competitive advantage through data-driven pricing and marketing.

**Key Questions Answered**:
- Where are the highest and lowest risk provinces and postal codes?
- Are there statistically significant risk differences by province, gender, and location?
- Can we predict claim occurrence and claim amount accurately?
- Which features most strongly influence risk and profitability?

---

## Key Findings (So Far)

| Insight | Evidence |
|-------|--------|
| Significant risk differences across provinces | Chi-squared p < 0.001 → **Reject null hypothesis** |
| Gauteng has highest claim frequency | 0.34% vs national 0.28% |
| Free State has highest average claim amount | R32,265 (vs Northern Cape R11,186) |
| Western Cape & Gauteng show negative average margin | Premiums < Claims → loss-making regions |
| Only 0.28% of policies result in a claim | Extremely imbalanced classification problem |
| Strong regional risk patterns detected | Province & postal code are powerful predictors |

---

## Project Structure
├── src/
│   ├── data_loader.py          # Load raw .txt or .csv
│   ├── cleaning.py             # Data cleaning & validation
│   ├── effects.py        
│   ├── equivalence.py             
│   ├── preprocessing.py        # Feature engineering & pipeline
│   ├── segmentation.py             
│   ├── statistical_tests.py      
│   ├── visualization.py             
│   ├── run_cleaning.py        
│   └── hypothesis_helper.py    # Pretty printing for statistical tests
├── notebooks/
│   ├── EDA_preprocessing.ipynb
│   ├── Task3_hypothesis_testing.ipynb
│   └── Task4_statistical_modeling.ipynb
├── data/
│   └── clean_data.csv          # Versioned with Git (or DVC later)
├── requirements.txt
└── README.md
text---

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