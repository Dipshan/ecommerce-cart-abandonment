# E-commerce Cart Abandonment Prediction & User Segmentation

This project analyzes over **109 million user events** from a large multi-category e-commerce store (October & November 2019) to:

1. Predict cart abandonment with a **balanced, production-ready ML model**, and  
2. Segment customers into **value- and behavior-based clusters** for marketing and product insights.

The pipeline is designed for **high-performance computing (HPC)** environments, combining:

- **Efficient CPU-based processing** for large-scale aggregation and clustering.
- **CPU-based Gradient Boosting (XGBoost / LightGBM)** and other ML models for prediction.

---

## 🚀 Project Goals

1. **Predict Abandonment**  
   Build a binary classifier to identify user sessions where a user **adds at least one item to cart but does not purchase**.

2. **Optimize for Balance (Precision vs Recall)**  
   Move beyond trivial "100% recall" strategies (flagging everyone) to a model with a **realistic balance** of:
   - High **Recall** for abandoners (catch the majority of at-risk users).
   - High **Precision** (minimize noise in interventions, such as coupons/emails).

3. **User Segmentation**  
   Cluster users by **behavioral shopping style** (e.g., browsers vs decisive buyers) to enable **actionable lifecycle marketing and personalization**.

---

## 📂 Project Structure

### 🐍 Core Pipeline Scripts

| File | Description |
| :--- | :--- |
| **`process_data_safe.py`** | **Step 1: Data Engineering (CPU).** Loads raw CSVs month-by-month to prevent memory overflows. Cleans events (e.g., `price > 0`), labels `is_abandoned`, engineers features (e.g., `is_returning_user`), and aggregates **109M+ events into ~23M sessions**. Saves `dataset/sessions_df_final_project.parquet`. |
| **`run_comprehensive_bakeoff_resumable.py`** | **Step 2: Model Training & Bake-Off (CPU).** Performs a 70/30 stratified split and runs a balanced bake-off between Logistic Regression, Random Forest, XGBoost, and LightGBM. Uses `GridSearchCV` to tune the top 2 models. **Resumable:** writes progress to `results/running_results.csv` so it can safely resume after interruptions. |
| **`train_and_save_champion.py`** | **Step 3: Champion Model (CPU).** Re-trains the best-performing model (tuned XGBoost) on the training data, computes feature importance, and saves the model (`champion_xgb_model.json`) and plots. |
| **`clustering.py`** | **Step 4: User Clustering (CPU).** Aggregates 23M+ sessions, engineers behavioral features, and performs K-Means clustering to identify 8 distinct user behaviors. Generates visualizations and detailed statistics. |
| **`EDA.ipynb`** | Exploratory Data Analysis notebook for initial dataset sanity checks and hypothesis exploration. |

### 📊 Results & Artifacts (`results/` folder)

| File / Pattern | Description |
| :--- | :--- |
| **`all_model_results.csv`** | Final leaderboard showing `weighted_avg_f1`, precision, recall, and training time for all candidate models. |
| **`running_results.csv`** | Intermediate results written by the resumable bake-off script. |
| **`champion_xgb_model.json`** | The saved, trained XGBoost model ready for deployment. |
| **`champion_feature_importance.png`** | Bar chart showing which features drive predictions (e.g., `num_carts`, `num_views`, `duration_seconds`). |
| **`*_confusion_matrix.png`** | Confusion matrix plots for all candidate models. |
| **`23m_event_cluster_analysis_detailed.csv`** | Detailed statistics for each of the 8 behavioral clusters. |
| **`23m_event_clustering_results_detailed.png`** | Comprehensive visualization dashboard showing cluster sizes, abandonment rates, and behaviors. |
| **`cluster_names_mapping.csv`** | Mapping of cluster IDs to intelligent names (e.g., "High-Risk", "Loyal Window Shopper"). |

---

## ⚙️ Setup & Installation

This project targets a **Python 3.10+** environment and was designed for an **HPC cluster** (e.g., 32 CPU cores, 64GB+ RAM).

### 1. Get the Data

This repository does **not** contain the raw data.

Download the following from the public [eCommerce Events History dataset on Kaggle](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store) (or a similar source):

- `2019-Oct.csv`
- `2019-Nov.csv`

Place them under:

```text
dataset/
  ├── 2019-Oct.csv
  └── 2019-Nov.csv
```

### 2. Create & Activate Virtual Environment (CPU Work)

```bash
python3 -m venv venv
source venv/bin/activate        # On macOS / Linux
# .\venv\Scripts\activate    # On Windows (PowerShell)
```

### 3. Install CPU Requirements (Modeling, EDA)

```bash
pip install     pandas     pyarrow     scikit-learn     xgboost     lightgbm     matplotlib     seaborn     jupyter
```



---

## 🏃‍♂️ Execution Instructions

> ⏱️ Full end-to-end run can take **several hours** depending on hardware, especially the model bake-off and clustering steps.

### 1. Data Processing (Step 1 — CPU, High RAM)

Aggregates 109M+ raw events into ~23M sessions and creates the final modeling dataset.

```bash
python3 process_data_safe.py
```

**Output:**

- `dataset/sessions_df_final_project.parquet`

### 2. Model Training & Bake-Off (Step 2 — CPU, Multi-Core)

Performs a 70/30 stratified split and runs a four-model bake-off.

```bash
# Recommended: run in the background on a server
nohup python3 run_comprehensive_bakeoff_resumable.py > training.log 2>&1 &
```

This script:

- Loads `sessions_df_final_project.parquet`.
- Trains Logistic Regression, Random Forest, XGBoost, and LightGBM with **no class weighting** for a *truly balanced baseline*.
- Ranks models by **weighted F1-score**.
- Selects the **Top 2** models and runs an extensive `GridSearchCV` on each.
- Saves **every run** into `results/running_results.csv` so progress is resumable.

**Output:**

- `results/all_model_results.csv`
- `results/*_confusion_matrix.png`
- `results/running_results.csv` (intermediate)

### 3. Train & Save Champion Model (Step 3 — CPU)

Retrains the best model and exports artifacts for deployment.

```bash
python3 train_and_save_champion.py
```

This script:

- Loads best hyperparameters and re-trains the **Champion** (XGBoost tuned).
- Saves the model to `results/champion_xgb_model.json`.
- Generates `results/champion_feature_importance.png`.

### 4. User Clustering & Segmentation (Step 4 — CPU)

Analyzes user behavior across 23M+ sessions to identify distinct shopping personas.

```bash
python3 clustering.py
```

This script:
- Loads optimized parquet data for October and November.
- Engineers session-level features (e.g., cart-to-view ratio, exploration depth).
- Performs **K-Means clustering** (k=8) to segment users.
- Automatically names clusters based on their characteristics (e.g., "High-Risk", "Loyal Window Shopper").
- Generates comprehensive visualizations and detailed statistics.

**Output:**
- `results/23m_event_clustering_results_detailed.png`
- `results/23m_event_cluster_analysis_detailed.csv`

---

## 🧠 Modeling Details & Final Leaderboard

### Target Definition

Each session is labeled as:

- `is_abandoned = 1` if the user **added at least one item to cart** but **did not complete a purchase** within that session.
- `is_abandoned = 0` otherwise.

### Feature Engineering Highlights

Key features include:

- `num_views` — Count of product view events within the session.
- `num_carts` — Count of add-to-cart events.
- `num_purchases` — Count of purchase events.
- `duration_seconds` — Session duration.
- `is_returning_user` — Whether the user has appeared in previous sessions.
- Several aggregate ratios (e.g., view/cart, cart/purchase) used downstream.

We also **deliberately removed** `max_price` after hypothesis testing showed it was noisy and did not improve performance.

### Final Model Leaderboard (Weighted F1, Sorted)

From `results/all_model_results.csv`:

| model_name          | weighted_avg_f1 | abandoned_recall | abandoned_precision | train_time_sec |
| :------------------ | --------------: | ---------------: | ------------------: | -------------: |
| **XGBoost_Tuned**   | **0.9697**      | **0.7986**       | **0.7150**          | **1054.27**    |
| XGBoost             | 0.9695          | 0.7932           | 0.7154              | 19.92          |
| LightGBM_Tuned      | 0.9693          | 0.7957           | 0.7117              | 31513.73       |
| LightGBM            | 0.9690          | 0.7942           | 0.7082              | 16.86          |
| RandomForest        | 0.9659          | 0.7491           | 0.6941              | 120.96         |
| LogisticRegression  | 0.9357          | 0.3035           | 0.5471              | 9.72           |

### Key Modeling Insights

1. **Champion Model: Tuned XGBoost**  
   - Recall (abandoners): **~80%** — identifies the majority of at-risk sessions.  
   - Precision (abandoners): **~71.5%** — high confidence when flagging likely abandoners.  
   - Overall: Excellent **weighted F1-score** and strong business viability.

2. **Cost vs Benefit of Tuning**  
   - Default XGBoost (~20 seconds) performs almost identically to Tuned XGBoost (~17.5 minutes).  
   - This suggests that **feature engineering quality** mattered more than hyperparameter tuning.

3. **Balanced, Actionable Model**  
   - The final model avoids pathological solutions (e.g., predict "abandoned" for everyone).  
   - It provides realistic trade-offs suitable for **real-world interventions** (discounts, reminders, UX changes).

4. **Feature Importance**  
   - `num_carts` is the dominant feature (~96% importance), effectively filtering **window shoppers**.  
   - `num_views` and `duration_seconds` are crucial for distinguishing **purchasers vs abandoners** once a cart exists.  
   - Removing `max_price` simplified the model and sped up training with **no loss in accuracy**.

---

## 👥 User Clustering & Segmentation

The project uses unsupervised learning to identify distinct user behaviors from the 23M+ sessions.

### Methodology

1. **Feature Engineering**: Extracts signals like session duration, event intensity, and conversion ratios.
2. **Clustering**: Uses **K-Means (k=8)** on robustly scaled features to group similar sessions.
3. **Intelligent Naming**: Automatically assigns descriptive names to clusters based on their statistical properties (e.g., abandonment rate, price sensitivity).

### Discovered Segments

The analysis typically identifies segments such as:

- **High-Risk**: Users with items in cart but very high abandonment rates.
- **Loyal Window Shoppers**: Frequent visitors who rarely buy.
- **Premium Shoppers**: Users interacting with high-value items.
- **Quick Browsers**: Short sessions with low engagement.

> 🔍 **View the Dashboard**: Check `results/23m_event_clustering_results_detailed.png` for a visual breakdown of all clusters.

---

## 🧩 How to Extend or Adapt This Project

A few ideas for next steps:

- **Real-Time Scoring**: Wrap `champion_xgb_model.json` in a REST API for live cart abandonment prediction.  
- **Uplift Modeling**: Estimate which users **change behavior** if targeted (vs those who would purchase anyway).  
- **Sequence Models**: Replace session-level features with **event sequences** (RNNs / Transformers).  
- **Multi-Channel Data**: Incorporate email, push, or ad interactions into user value and behavioral features.

---

## 📜 License & Attribution

- Data originates from the public **eCommerce behavior dataset** (see Kaggle listing for license and citation details).
- Model code, feature engineering, and clustering pipeline were built for research/educational purposes and can be adapted to your own e-commerce data (subject to your org's internal policies).

If you use or extend this project, consider citing it in your work and sharing improvements via pull requests or issues.

Happy modeling! 🛒📈
