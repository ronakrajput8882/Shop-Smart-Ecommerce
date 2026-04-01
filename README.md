<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=2,12,24&height=200&section=header&text=🛒%20ShopSmart%20Purchase%20Intent%20Predictor&fontSize=38&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Predict%20Online%20Shopper%20Purchasing%20Intent%20with%20Decision%20Tree%20%7C%20Sklearn%20Pipeline&descAlignY=60&descAlign=50" width="100%"/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

---

## 📌 Project Overview

E-commerce businesses lose millions every year to cart abandonment and missed conversions. This project builds a **binary classification model** that predicts whether an online shopping session will end in a purchase — using behavioral signals like page views, bounce rates, exit rates, and session metadata.

The goal: give e-commerce teams a reliable signal to personalize UX or trigger real-time interventions for high-intent visitors.

- **Task:** Binary Classification (`Revenue`: Purchase or No Purchase)
- **Dataset:** Online Shoppers Purchasing Intention Dataset
- **Best Model:** Decision Tree Classifier (Tuned via GridSearchCV)

---

## 📂 Dataset

| Property | Value |
|:---|:---|
| Total Samples | 12,329 |
| Features | 17 (10 numerical, 7 categorical) |
| Target | `Revenue` (True/False → 1/0) |
| Class Imbalance | ~84.5% No Purchase / ~15.5% Purchase |
| Missing Values | None |

**Key features:**
- `PageValues` — average value of pages visited before transaction
- `ExitRates` — average exit rate of pages visited
- `BounceRates` — average bounce rate of pages visited
- `ProductRelated`, `ProductRelated_Duration` — product page engagement
- `Month`, `VisitorType`, `Weekend` — session context

---

## 🔄 Pipeline Workflow

```
Raw CSV → EDA → Feature Split → Train/Test Split → Preprocessing Pipeline → Model Training → GridSearchCV → Evaluation
```

1️⃣ **Load & Inspect** — Load CSV, check nulls, understand class distribution  
2️⃣ **Feature Separation** — Split numerical vs categorical columns automatically using `select_dtypes`  
3️⃣ **Train/Test Split** — 80/20 split with `stratify=y` to preserve class ratios  
4️⃣ **Preprocessing Pipeline** — `StandardScaler` for numerics, `OneHotEncoder` for categoricals via `ColumnTransformer`  
5️⃣ **Model Training** — `DecisionTreeClassifier` with `class_weight="balanced"` to handle imbalance  
6️⃣ **Hyperparameter Tuning** — `GridSearchCV` over `max_depth` and `min_samples_leaf`  
7️⃣ **Evaluation** — F1 Score, Precision, Recall, Classification Report, Confusion Matrix  

---

## 🤖 Models

### 1️⃣ Decision Tree Classifier ⭐ Best Model

```python
dtc_model = DecisionTreeClassifier(
    max_depth=6,
    min_samples_leaf=30,
    class_weight="balanced",
    random_state=42
)

pipe = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", dtc_model)
])
```

- `class_weight="balanced"` corrects for the 84/16 class imbalance automatically
- `max_depth=6` prevents overfitting on noisy behavioral signals
- `min_samples_leaf=30` ensures stable leaf nodes with sufficient support
- Best GridSearchCV params: `max_depth=4`, `min_samples_leaf=50`

---

## 📊 Results

| Model | Precision | Recall | F1 Score | Accuracy |
|:---|:---:|:---:|:---:|:---:|
| Decision Tree (Baseline) | 0.504 | 0.833 | 0.628 | 84.71% |
| 🏆 **Decision Tree (Tuned)** | **0.498** | **0.833** | **0.623** | **84.39%** |

> **Note:** High recall (83.3%) is the key optimization target — missing a real buyer (False Negative) costs more than flagging a non-buyer (False Positive). The `class_weight="balanced"` parameter drives this recall-first behavior.

---

## 🔍 Key Insights

- 📈 **`PageValues`** has a **+0.49 positive correlation** with Revenue — the single strongest predictor; users who browse high-value pages are far more likely to purchase
- 📉 **`ExitRates`** has a **-0.21 negative correlation** with Revenue — high exit rates are a clear signal of abandonment intent
- 📉 **`BounceRates`** has a **-0.15 negative correlation** with Revenue — single-page sessions rarely convert
- **`ProductRelated` page engagement** (both count and duration) correlates positively at +0.15, confirming deeper product browsing = higher purchase probability
- `SpecialDay` proximity slightly reduces purchase probability (-0.08) — possibly due to browsing-only behavior during holidays
- **`class_weight="balanced"`** was critical: without it, the model predicts mostly non-purchase due to 84/16 class split

---

## 🗂️ Repository Structure

```
shop-smart-ecommerce/
│
├── shop_smart_ecommerce.ipynb   # Main notebook: EDA, training, evaluation
├── shop_smart_ecommerce.csv     # Dataset (Online Shoppers Purchasing Intention)
└── README.md                    # Project documentation
```

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/ronakrajput8882/Shop-Smart-Ecommerce.git
cd Shop-Smart-Ecommerce

# Install dependencies
pip install pandas scikit-learn seaborn matplotlib jupyter

# Launch notebook
jupyter notebook shop_smart_ecommerce.ipynb
```

---

## 🧠 Key Learnings

- **Imbalanced classification requires intentional design** — `class_weight="balanced"` shifted the model from 0% recall on the minority class to 83%, without any oversampling
- **Sklearn Pipelines prevent data leakage** — fitting the scaler only on train data, applied on test, is the correct production pattern
- **High recall ≠ high precision in imbalanced tasks** — depending on business cost, optimizing for recall over F1 or accuracy may be the right call
- **`PageValues` dominates** — a single engineered feature derived from Google Analytics carries more signal than all session-time features combined
- **GridSearchCV with `scoring="f1"`** ensures tuning aligns with actual business metric, not just accuracy

---

## 🛠️ Tech Stack

| Tool | Use |
|:---|:---|
| Python 3.10+ | Core language |
| Pandas | Data loading & manipulation |
| Scikit-learn | Pipeline, preprocessing, model, GridSearch |
| Seaborn / Matplotlib | EDA visualization |
| Jupyter Notebook | Interactive development |

---

<div align="center">

### Connect with me

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/ronakrajput8882)
[![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://instagram.com/techwithronak)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ronakrajput8882)

*If you found this useful, please ⭐ the repo!!!*

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=2,12,24&height=100&section=footer" width="100%"/>

</div>
