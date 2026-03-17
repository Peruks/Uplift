# 📊 Customer Segmentation, A/B Testing & Uplift Modeling
### From Average Experiments to Personalized Decisions

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Case Study](https://img.shields.io/badge/Case%20Study-Live%20Demo-purple)](https://upliftmodeling.lovable.app/)

> This project demonstrates how traditional A/B testing can fall short in real-world marketing scenarios — and how **uplift modeling** enables smarter, customer-level decisions instead of relying on average-based effects.

---

## 📌 Problem Statement

Marketing teams commonly use A/B testing to evaluate campaign performance. However, A/B testing measures the **average treatment effect**, which assumes all customers respond similarly.

In reality:

| Customer Type | Response to Treatment |
|---|---|
| 🟢 Persuadables | Respond positively — the ideal target |
| ⚪ Sure Things | Convert regardless — no need to spend |
| 🔴 Do Not Disturbs | Respond negatively — avoid targeting |
| ⚫ Lost Causes | Unlikely to convert either way |

When these effects are averaged together, they cancel out, leading to:

```
❌ "No statistical significance"
❌ Missed business opportunities
```

---

## 🎯 Objective

Move from a population-level question:
> *"Which variant performs better on average?"*

To a customer-level question:
> *"Which specific customers should receive the treatment?"*

---

## 📂 Dataset

**Bank Marketing Dataset** — [UCI / Kaggle](https://archive.ics.uci.edu/ml/datasets/bank+marketing)

| Property | Details |
|---|---|
| Records | ~41,000 customer entries |
| Target Variable | `y` — term deposit subscription (yes/no) |

**Feature Categories:**

- **Demographics** — age, job, marital status, education
- **Financial** — loan status, default history, balance
- **Campaign Behavior** — contact type, number of contacts, call duration
- **Macro-economic Indicators** — employment rate, consumer price index, euribor rate

---

## ⚠️ Data Leakage Handling

Certain features (`pdays`, `poutcome`) contain information about past campaign outcomes, which would leak future information into the model.

**Solution — engineer a safe binary feature:**

```python
df['contacted_before'] = (df['pdays'] != 999).astype(int)
```

This preserves the signal (whether the customer was previously contacted) without exposing the outcome of that contact.

---

## 🧹 Data Preprocessing

```python
# Binary target encoding
df['converted'] = (df['y'] == 'yes').astype(int)

# One-hot encoding for categorical variables
df = pd.get_dummies(df, drop_first=True)

# Feature scaling (required for K-Means)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

## 🔍 Customer Segmentation (K-Means)

Customers were segmented using **K-Means clustering** to identify distinct behavioral groups before running experiments — providing a foundation for detecting heterogeneous treatment effects.

### Choosing the Optimal K

Two complementary methods were used:

```python
from sklearn.metrics import silhouette_score

inertias, silhouettes = [], []
for k in range(2, 10):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, km.labels_))
```

| Method | Signal |
|---|---|
| Elbow Method | Inertia drop rate |
| Silhouette Score | Cluster cohesion & separation |

**Final choice: K = 3**

### Cluster Insights

The three segments showed meaningfully different behaviors across:
- Campaign engagement frequency
- Loan and default status
- Previous contact history

This variation strongly suggested **heterogeneous treatment effects** across segments — a key motivation for uplift modeling.

---

## 🧪 A/B Testing

### Experimental Setup

```python
import numpy as np

# Random assignment to control/treatment
df['variant'] = np.where(np.random.rand(len(df)) < 0.5, 'A', 'B')

control   = df[df['variant'] == 'A']['converted']
treatment = df[df['variant'] == 'B']['converted']
```

### Statistical Test

A **two-proportion Z-test** was used to assess whether conversion rates differed significantly between variants.

```python
from statsmodels.stats.proportion import proportions_ztest

count = [treatment.sum(), control.sum()]
nobs  = [len(treatment), len(control)]
z_stat, p_value = proportions_ztest(count, nobs)
```

### Result

```
p-value > 0.05
→ Fail to reject the null hypothesis
→ No statistically significant difference detected
```

### ⚠️ Key Insight

The experiment produced a **non-significant result** — not because the treatment had no effect, but because **averaging across heterogeneous customer groups masked the real signal**.

---

## 🧠 Heterogeneous Treatment Effects

Breaking down the A/B test results by customer segment revealed the root cause:

| Segment | Treatment Effect |
|---|---|
| Segment 0 | ✅ Positive response |
| Segment 1 | ➖ Neutral / no effect |
| Segment 2 | ❌ Negative response |

The positive and negative effects cancelled each other out at the population level — explaining why the aggregate result was insignificant.

---

## 🔄 Uplift Modeling

### Approach: T-Learner

Two separate models were trained — one on the control group, one on the treatment group:

```python
from sklearn.ensemble import GradientBoostingClassifier

# Train control model
model_control = GradientBoostingClassifier(random_state=42)
model_control.fit(X_control, y_control)

# Train treatment model
model_treatment = GradientBoostingClassifier(random_state=42)
model_treatment.fit(X_treatment, y_treatment)
```

### Uplift Calculation

```python
# Estimate individual-level treatment effect
uplift = model_treatment.predict_proba(X)[:, 1] \
       - model_control.predict_proba(X)[:, 1]

uplift_df['predicted_uplift'] = uplift
```

This estimates **individual causal impact** — how much more (or less) likely each customer is to convert if treated.

---

## 🎯 Targeting Strategy

Customers were ranked by predicted uplift. Only the **top 20%** (highest uplift) were selected for targeting:

```python
threshold = uplift_df['predicted_uplift'].quantile(0.80)
top_20_pct = uplift_df[uplift_df['predicted_uplift'] >= threshold]

print(f"Targeted customers : {len(top_20_pct)}")
print(f"Avg predicted uplift: {top_20_pct['predicted_uplift'].mean():.4f}")
```

**Outcome:**
- ✅ Focus spend on customers most likely to respond
- 🚫 Avoid customers who react negatively (Do Not Disturbs)
- 📈 Improve campaign ROI through precision targeting

---

## 📊 Results Summary

| Approach | Decision Basis | Outcome |
|---|---|---|
| A/B Testing | Average treatment effect | No significance detected |
| Uplift Modeling | Individual treatment effect | Actionable customer segments identified |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `scikit-learn` | Clustering, ML models, preprocessing |
| `statsmodels` | Z-test for A/B testing |
| `matplotlib`, `seaborn` | Visualization |
| `Python 3.8+` | Core language |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/perarivalanks/uplift-modeling.git
cd uplift-modeling
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Notebook

```bash
jupyter notebook uplift_modeling.ipynb
```

---

## 📁 Project Structure

```
uplift-modeling/
│
├── data/
│   └── bank-additional-full.csv    # Raw dataset
│
├── notebooks/
│   └── uplift_modeling.ipynb       # Main analysis notebook
│
├── src/
│   ├── preprocessing.py            # Data cleaning & feature engineering
│   ├── segmentation.py             # K-Means clustering
│   ├── ab_testing.py               # A/B test & Z-test
│   └── uplift.py                   # T-Learner uplift model
│
├── requirements.txt
└── README.md
```

---

## 🔮 Future Improvements

- [ ] **Power Analysis** — determine required sample size before experimentation
- [ ] **X-Learner & Causal Forests** — more robust uplift models for high-dimensional data
- [ ] **AUUC Curve** — proper uplift model evaluation metric
- [ ] **Real-time Scoring** — deploy model as a REST API for live campaign decisions
- [ ] **Streamlit Dashboard** — business-friendly UI for non-technical stakeholders

---

## 📚 References

- Gutierrez, P., & Gérardy, J. Y. (2017). *Causal Inference and Uplift Modeling: A Review of the Literature*
- [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- Künzel et al. (2019). *Metalearners for Estimating Heterogeneous Treatment Effects*

---

## 👨‍💻 Author

**Perarivalan** —  Data Scientist & AI Engineer Practioner

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/perarivalanks/)
[![Case Study](https://img.shields.io/badge/Case%20Study-Live%20Demo-purple)](https://upliftmodeling.lovable.app/)

---

> *"No statistical significance does not mean no opportunity — it means you need a better model."*
