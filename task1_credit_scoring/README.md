# Task 1: Credit Scoring Model

## Objective
Predict whether a credit card customer will default on payment next month (1 = default).

## Dataset
UCI "Default of Credit Card Clients" (Taiwan, 2005): https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
- 30000 customers after removing duplicates
- 30 features (23 original + 7 engineered)
- Default rate: 22.1% (imbalanced, so accuracy alone is misleading)
- File used: `default of credit card clients.xls` (loaded with `header=1`)

## Approach
1. Cleaned undocumented category codes, removed duplicates, dropped the ID column
2. EDA: default rate by payment status, demographics, correlations
3. Feature engineering: avg_delay, max_delay, months_late, avg_bill, total_paid, utilization, pay_ratio
4. Preprocessing: one-hot encoding + scaling inside a Pipeline (no data leakage)
5. Stratified 80/20 split (24000 train / 6000 test)
6. Models: Logistic Regression, Decision Tree, Random Forest (plain and class-balanced)
7. Evaluation: Precision, Recall, F1, ROC-AUC, 5-fold cross-validation

## Results (test set)
| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.637 | 0.295 | 0.404 | 0.744 |
| Decision Tree | 0.675 | 0.346 | 0.457 | 0.759 |
| Random Forest | 0.653 | 0.353 | 0.459 | 0.777 |
| LogReg (balanced) | 0.44 | 0.601 | 0.508 | 0.747 |
| RandomForest (balanced) | 0.483 | 0.602 | 0.536 | 0.777 |

Best ROC-AUC: **Random Forest** (0.777).
5-fold CV ROC-AUC of the balanced Random Forest: 0.784 ± 0.005.

## Threshold analysis (balanced Random Forest)
| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.3 | 0.312 | 0.858 | 0.457 |
| 0.4 | 0.397 | 0.698 | 0.506 |
| 0.5 | 0.483 | 0.602 | 0.536 |
| 0.6 | 0.563 | 0.518 | 0.539 |

Moving the threshold from 0.6 to 0.3 changes recall from 0.518 to 0.858 and precision from 0.563 to 0.312. A bank that wants to catch more defaulters accepts lower precision.

Top 5 features (Random Forest): max_delay, PAY_0, avg_delay, months_late, PAY_2.

![ROC curves](images/roc_curves.png)
![Confusion matrix](images/confusion_matrix.png)
![Feature importance](images/feature_importance.png)

## Limitations
- The threshold was compared on the test set; a separate validation set would be cleaner.
- Data is from 2005 and from one country.

## Demo app (optional)
```
python -m streamlit run app.py
```
A small Streamlit form that estimates default probability. It approximates six months of history using one average bill and one average payment.

## How to run
Open `credit_scoring.ipynb` and click Run All.
