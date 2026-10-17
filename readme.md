# Task 1: Credit Scoring Model

## Objective
Predict whether a credit card customer will default next month.

## Dataset
UCI "Default of Credit Card Clients" (30,000 customers, Taiwan).
Source: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
Target: `default` (1 = default). About [X]% of customers default (class imbalance).

## Approach
1. Cleaned undocumented category codes, removed duplicates
2. EDA: default rate by payment status, demographics, correlations
3. Feature engineering: avg_delay, max_delay, months_late, utilization, pay_ratio
4. Preprocessing: one-hot encoding + scaling inside a Pipeline
5. Models: Logistic Regression, Decision Tree, Random Forest (plain and class-balanced)
6. Evaluation: Precision, Recall, F1, ROC-AUC, 5-fold cross-validation

## Results (test set)
| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| [paste your rows] | | | | |

Best model: [name]. Threshold analysis: [one sentence on precision vs recall].

![ROC](images/roc_curves.png)
![Features](images/feature_importance.png)

## Limitations
- Threshold was chosen using the test set; a validation set would be cleaner.
- Data is from 2005 and from one country.

## How to run
pip install -r ../requirements.txt
Open credit_scoring.ipynb and Run All.