# E-commerce Cart Abandonment Prediction & User Segmentation

This project analyzes over **109 million user events** from a large multi-category e-commerce store (October & November 2019) to:

1. Predict cart abandonment with a **balanced, production-ready ML model**, and  
2. Segment customers into **value- and behavior-based clusters** for marketing and product insights.

The pipeline is designed for **high-performance computing (HPC)** environments, combining:

- **GPU acceleration (RAPIDS/cuDF)** for large-scale aggregation and clustering.
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
   Cluster users by:
   - **RFM (Recency, Frequency, Monetary) value**, and  
   - **Behavioral shopping style** (e.g., browsers vs decisive buyers)  
   to enable **actionable lifecycle marketing and personalization**.

---

## 📂 Project Structure

### 🐍 Core Pipeline Scripts

| File | Description |
| :--- | :--- |
| **`process_data_safe.py`** | **Step 1: Data Engineering (CPU).** Loads raw CSVs month-by-month to prevent memory overflows. Cleans events (e.g., `price > 0`), labels `is_abandoned`, engineers features (e.g., `is_returning_user`), and aggregates **109M+ events into ~23M sessions**. Saves `dataset/sessions_df_final_project.parquet`. |
| **`run_comprehensive_bakeoff_resumable.py`** | **Step 2: Model Training & Bake-Off (CPU).** Performs a 70/30 stratified split and runs a balanced bake-off between Logistic Regression, Random Forest, XGBoost, and LightGBM. Uses `GridSearchCV` to tune the top 2 models. **Resumable:** writes progress to `results/running_results.csv` so it can safely resume after interruptions. |
| **`train_and_save_champion.py`** | **Step 3: Champion Model (CPU).** Re-trains the best-performing model (tuned XGBoost) on the training data, computes feature importance, and saves the model (`champion_xgb_model.json`) and plots. |
| **`run_user_clustering_unified.py`** | **Step 4: User Clustering (GPU + CPU).** Aggregates all sessions into **user-level profiles** (5.3M+ users) using **RAPIDS cuDF** on the GPU, then runs **MiniBatchKMeans** (CPU) to generate RFM and behavioral clusters. |
| **`EDA.ipynb`** | Exploratory Data Analysis notebook for initial dataset sanity checks and hypothesis exploration. |

### 📊 Results & Artifacts (`results/` folder)

| File / Pattern | Description |
| :--- | :--- |
| **`all_model_results.csv`** | Final leaderboard showing `weighted_avg_f1`, precision, recall, and training time for all candidate models. |
| **`running_results.csv`** | Intermediate results written by the resumable bake-off script. |
| **`champion_xgb_model.json`** | The saved, trained XGBoost model ready for deployment. |
| **`champion_feature_importance.png`** | Bar chart showing which features drive predictions (e.g., `num_carts`, `num_views`, `duration_seconds`). |
| **`*_confusion_matrix.png`** | Confusion matrix plots for all candidate models. |
| **`cluster_profile_rfm.csv`** | Average statistics for each **RFM (value-based)** user cluster. |
| **`cluster_profile_style.csv`** | Average statistics for each **behavioral (shopping-style)** user cluster. |
| **`*_clusters.png`** | Visualization plots (e.g., 2D projections or bar charts) summarizing the discovered segments. |

---

## ⚙️ Setup & Installation

This project targets a **Python 3.10+** environment and was designed for an **HPC cluster** (e.g., 32 CPU cores, 64GB+ RAM) with **optional NVIDIA GPU** for clustering.

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

### 4. Install GPU Requirements (Optional: RAPIDS for Clustering)

GPU acceleration is **optional but strongly recommended** for the user clustering step.

You will need:

- Recent NVIDIA GPU
- Compatible NVIDIA driver
- CUDA 11/12

Recommended: use **Conda** to create a RAPIDS environment (example for CUDA 12):

```bash
conda create -n rapids-env python=3.10
conda activate rapids-env

conda install -c rapidsai -c conda-forge -c nvidia     rapids=24.10     cuda-version=12.0
```

> 💡 If RAPIDS is not available on your system, you can modify `run_user_clustering_unified.py` to use pure pandas instead of cuDF (at the cost of runtime).

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

### 4. User Clustering & Segmentation (Step 4 — GPU + CPU)

Builds user-level profiles and segments customers into RFM and behavioral clusters.

```bash
python3 run_user_clustering_unified.py
```

This script:

- Uses **RAPIDS cuDF** to rapidly aggregate all sessions to user-level stats.
- Uses **MiniBatchKMeans** to cluster users on:
  - **RFM (Recency, Frequency, Monetary) features**, and
  - **Shopping-style features** (view/cart/purchase ratios).
- Exports cluster profiles and plots to the `results/` directory.

**Output:**

- `results/cluster_profile_rfm.csv`
- `results/cluster_profile_style.csv`
- `results/*_clusters.png`

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

Beyond per-session predictions, the project builds **user-level segments** to inform marketing and personalization strategies.

### Methodology

1. Aggregate 23M+ sessions into **5.3M+ user profiles**.
2. For each user, compute:
   - RFM metrics (Recency of last visit, total Frequency, total Monetary value).
   - Behavioral metrics (e.g., views/session, carts/session, cart-to-purchase conversion).
3. Run **MiniBatchKMeans** separately for:
   - **RFM clustering** (value-based segments).
   - **Shopping-style clustering** (behavioral segments).

### 1️⃣ RFM Clusters (Value-Based Personas)

Representative segments (exact thresholds in `cluster_profile_rfm.csv`):

- **Champions**  
  - High spend (e.g., **$764+ average**).  
  - High visit frequency.  
  - Extremely valuable for retention and VIP programs.

- **Loyal Window Shoppers**  
  - Visit often (e.g., **9+ visits**) but spend near zero.  
  - Likely exploring or price-sensitive; candidates for **targeted discounts** or **UX improvements**.

- **Churned**  
  - Have not visited in **40+ days**.  
  - Strong candidates for **win-back campaigns** (email, push, ads).

- **Casual**  
  - Recent visitors with low to moderate activity and spend.  
  - Perfect for **onboarding journeys** and general nudges.

### 2️⃣ Shopping Style Clusters (Behavior-Based Personas)

Representative segments (exact stats in `cluster_profile_style.csv`):

- **Decisive Shoppers**  
  - High **view-to-cart ratio** (~0.60).  
  - Find what they want quickly and convert efficiently.

- **Browsers**  
  - Very high **views per session** (e.g., 11+), but relatively low conversion.  
  - Might be comparing products or exploring categories — ideal for stronger **recommendation systems** and **guided search**.

- **Power Buyers**  
  - Extremely high **cart-to-purchase conversion rate**.  
  - Excellent target for **loyalty programs**, **upselling**, and **early access** campaigns.

> 🔍 All cluster statistics and exact thresholds are defined in:
> - `results/cluster_profile_rfm.csv`
> - `results/cluster_profile_style.csv`

---

## 🧩 How to Extend or Adapt This Project

A few ideas for next steps:

- **Real-Time Scoring**: Wrap `champion_xgb_model.json` in a REST API for live cart abandonment prediction.  
- **Uplift Modeling**: Estimate which users **change behavior** if targeted (vs those who would purchase anyway).  
- **Sequence Models**: Replace session-level features with **event sequences** (RNNs / Transformers).  
- **Multi-Channel Data**: Incorporate email, push, or ad interactions into RFM and behavioral features.

---

## 📜 License & Attribution

- Data originates from the public **eCommerce behavior dataset** (see Kaggle listing for license and citation details).
- Model code, feature engineering, and clustering pipeline were built for research/educational purposes and can be adapted to your own e-commerce data (subject to your org's internal policies).

If you use or extend this project, consider citing it in your work and sharing improvements via pull requests or issues.

Happy modeling! 🛒📈
