# Titanic Survival Prediction

A binary classification project predicting passenger survival on the Titanic, using the classic [Kaggle Titanic dataset](https://www.kaggle.com/c/titanic). Built from scratch as an end-to-end workflow: EDA → preprocessing → feature engineering → model comparison → Kaggle submission.

## Problem

Given passenger information (age, sex, ticket class, fare, etc.), predict whether a passenger survived (`1`) or not (`0`).

## Dataset

- **Source:** Kaggle Titanic competition
- **Training samples:** 891 passengers
- **Target:** `Survived` (0 = died, 62%, 1 = survived, 38%)
- **Features:** ticket class, sex, age, siblings/spouses aboard, parents/children aboard, fare, port of embarkation

## Approach

**1. Exploratory Data Analysis**
- Inspected distributions, missing values, and feature-target correlations
- Confirmed `Fare` correlates with survival (0.25), so it was kept rather than dropped

**2. Preprocessing & Feature Engineering**
- Dropped `PassengerId` (identifier) and `Cabin` (77% missing)
- **Extracted titles** (`Mr`, `Miss`, `Mrs`, `Master`, `Rare`) from the `Name` field — a compact proxy for age, sex, and social status
- **Title-based age imputation:** filled missing `Age` values using the median age *per title* (e.g. `Master` → child age, `Mrs` → adult age) instead of a single global median
- Encoded categorical variables (`Sex`, `Embarked`, `Title`)

**3. Model Comparison**
Compared 7 classifiers on accuracy, precision, recall, and F1:

| Model | Accuracy | F1 |
|-------|----------|-----|
| **Gradient Boosting** | **0.838** | **0.836** |
| Random Forest | 0.827 | 0.826 |
| AdaBoost | 0.799 | 0.798 |
| XGBoost | 0.793 | 0.794 |
| Hist Gradient Boosting | 0.788 | 0.787 |
| Decision Tree | 0.771 | 0.771 |
| KNN | 0.682 | 0.678 |

## Key Findings

- **Feature engineering mattered more than model choice.** Extracting titles and using title-based age imputation raised the Kaggle score from **0.710 → 0.794** — an 8-point jump with the same underlying models.
- **Title captures information that age + sex alone miss** (marital status, social standing) and also compensates for missing `Age` values.
- **The best model changed with the features.** XGBoost led before feature engineering but Gradient Boosting led after, showing why comparing multiple models matters.
- Accuracy ≈ F1 here because the classes are only mildly imbalanced (unlike heavily skewed datasets where accuracy is misleading).

## Latest Kaggle Result

| Submission | Public Score |
|-----------|--------------|
| Baseline (no feature engineering) | 0.710 |
| **+ Title extraction & title-based age imputation** | **0.794** |

## Tech Stack

pandas, numpy, scikit-learn, xgboost

## Project Structure

```
titanic/
├── EDA.ipynb          # EDA, preprocessing, feature engineering, model comparison
├── Prediction.ipynb   # Test set prediction & submission generation
└── README.md
```

## Getting Started

```bash
pip install pandas numpy scikit-learn xgboost
jupyter notebook EDA.ipynb
```

Download `train.csv` and `test.csv` from the [Kaggle Titanic competition](https://www.kaggle.com/c/titanic/data) into the project folder.
