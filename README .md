# Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions using Logistic Regression and Random Forest classifiers. The dataset is highly imbalanced (~0.17% fraud), making this a realistic and challenging classification problem.

---

## Problem Statement

Credit card fraud causes billions in losses annually. The core challenge here is not just building a classifier — it's building one that performs well on an extremely rare class (fraud), where standard accuracy metrics are misleading and the cost of missing a fraud far outweighs a false alarm.

---

## Dataset

- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions, 492 fraudulent (0.17%)
- **Features:** V1–V28 (PCA-transformed), `Time`, `Amount`, `Class` (target)

---

## Project Structure

```
credit-card-fraud/
├── README.md
├── notebook.ipynb
├── src/
│   ├── preprocess.py
│   └── model.py
└── results/
    └── roc_curve.png
```

---

## Approach

### 1. Exploratory Data Analysis
- Visualized class imbalance
- Analyzed distribution of `Amount` and `Time` across fraud vs. legit transactions

### 2. Preprocessing
- Scaled `Amount` and `Time` using `StandardScaler` (V1–V28 are already PCA-transformed)
- Applied **SMOTE** (Synthetic Minority Oversampling Technique) to address class imbalance on the training set only

### 3. Models

| Model | Why Used |
|---|---|
| Logistic Regression | Strong interpretable baseline; useful for understanding feature weights |
| Random Forest | Handles non-linear fraud patterns better; more robust on imbalanced data |

### 4. Evaluation

Standard accuracy is not meaningful here — a model predicting "legit" every time scores 99.8% accuracy but catches zero fraud.

Metrics used:
- **Precision** — of all flagged transactions, how many were actually fraud?
- **Recall** — of all actual frauds, how many did we catch?
- **F1 Score** — harmonic mean of precision and recall; best single metric for imbalanced datasets
- **AUC-ROC** — measures model's ability to distinguish between classes across all thresholds

---

## Results

| Model | F1 Score (Fraud) | AUC-ROC |
|---|---|---|
| Logistic Regression | — | — |
| Random Forest | — | — |

> Results will be updated after final runs.

---

## Key Findings

- Logistic Regression provides a strong, interpretable baseline but is limited by the assumption of linear decision boundaries — fraud patterns in this dataset are non-linear.
- Random Forest achieves a higher F1 score and AUC-ROC on the minority class, making it the better production choice where catching fraud matters most.
- SMOTE significantly improved recall on both models by preventing the classifier from ignoring the minority class entirely.

---

## How to Run

```bash
git clone https://github.com/arnavgoel1196/credit-card-fraud.git
cd credit-card-fraud
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

---

## Dependencies

```
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
seaborn
jupyter
```

---

## Author

**Arnav Goel**  
[GitHub](https://github.com/arnavgoel1196) · [LinkedIn](#)
