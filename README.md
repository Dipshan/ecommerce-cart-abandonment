# E-commerce Cart Abandonment Prediction

This project analyzes over 100 million user events from a large e-commerce store to predict cart abandonment. The goal is to build a *balanced* and *useful* machine learning model that provides a high-performing, realistic balance between precision and recall.

---

## 🚀 Project Goal

The primary objective is to build a model that can accurately identify users who are likely to abandon their shopping carts. This involved:
1.  **Data Engineering:** Processing over 109 million raw event-level records from October and November 2019 into 23 million aggregated user sessions.
2.  **Feature Engineering:** Creating new, high-value features, such as `is_returning_user`.
3.  **Hypothesis Testing:** Confirming that the `max_price` feature was "noise" and that an 8-feature model would be more effective.
4.  **Model Bake-Off:** Running a comprehensive, balanced bake-off between Logistic Regression, Random Forest, XGBoost, and LightGBM to find the true champion model.

---

## 📁 Project Structure

This repository contains the following key files:

* `process_data_safe.py`: **(Step 1)** The main data processing pipeline. This script:
    * Loads the raw `.csv` files month by month (to prevent memory crashes).
    * Cleans the data (e.g., `price > 0`).
    * Creates the `is_abandoned` target label.
    * Aggregates 109M+ events into 23M user sessions.
    * Engineers all features, including `is_returning_user`.
    * Saves the final, model-ready `dataset/sessions_df_final_project.parquet` file.

* `run_comprehensive_bakeoff_resumable.py`: **(Step 2)** The main modeling pipeline. This script:
    * Loads the final parquet file.
    * Performs a 70/30 train-test split.
    * Runs a "default" bake-off between four models (LR, RF, XGB, LGBM) with **no class weighting** to get a true balanced baseline.
    * Identifies the Top 2 models based on their `weighted_avg_f1` score.
    * Runs an extensive, multi-hour `GridSearchCV` on the top 2 models.
    * This script is **resumable**: if it's stopped, it will read the `running_results.csv` and pick up where it left off.

* `EDA.ipynb`: A Jupyter Notebook used for initial exploratory data analysis.

* `/results/`: This folder contains all the final outputs from the modeling script.
    * `all_model_results.csv`: The final, sorted leaderboard comparing all 6 models (4 default, 2 tuned).
    * `*.png`: All confusion matrix plots for each model.

---

## ⚙️ How to Run This Project

This project requires a high-performance machine (e.g., a 32-core, 64GB RAM cluster) and will take several hours to run.

### 1. Setup

1.  **Get the Data:** This repository does **not** contain the raw data. You must acquire `2019-Oct.csv` and `2019-Nov.csv` and place them in a `dataset/` folder.
2.  **Create a Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install Requirements:**
    ```bash
    pip install pandas pyarrow scikit-learn xgboost lightgbm matplotlib seaborn jupyter
    ```

### 2. Step 1: Process the Data

Run the first script to process all 109M+ events. This will take 20-30 minutes.

```bash
python3 process_data_safe.py
```
**Output:** This creates the `dataset/sessions_df_final_project.parquet` file.

### 3. Step 2: Run the Model Bake-Off

Run the second script to train all the models. This will take **9-10 hours**. It is highly recommended to run this using `nohup` on a server.

```bash
nohup python3 run_comprehensive_bakeoff_resumable.py > script.log 2>&1 &
```
**Output:** This will populate the `results/` folder with all plots and the final `all_model_results.csv`.

---

## 🏆 Final Results & Key Findings

The full model comparison is in `results/all_model_results.csv`.

### Final Model Leaderboard (Sorted by Weighted F1-Score)

| model_name | weighted_avg_f1 | abandoned_recall | abandoned_precision | train_time_sec |
| :--- | :--- | :--- | :--- | :--- |
| **XGBoost_Tuned** | **0.9697** | **0.7986** | **0.7150** | **1054.27** |
| XGBoost | 0.9695 | 0.7932 | 0.7154 | 19.92 |
| LightGBM_Tuned | 0.9693 | 0.7957 | 0.7117 | 31513.73 |
| LightGBM | 0.9690 | 0.7942 | 0.7082 | 16.86 |
| RandomForest | 0.9659 | 0.7491 | 0.6941 | 120.96 |
| LogisticRegression | 0.9357 | 0.3035 | 0.5471 | 9.72 |

### 💡 Analysis

1.  **Champion Model:** The `XGBoost_Tuned` model is the clear winner, providing the best balance of precision and recall. It successfully identifies **80%** of abandoning users, and when it makes a prediction, it is correct **71.5%** of the time.

2.  **The "Cost" of Tuning:** The default `XGBoost` model (19.9 seconds) performed almost *identically* to the tuned version (17.5 minutes). This is a major finding: the **default parameters are already excellent**, and our feature engineering was the most important factor.

3.  **Balanced Model Success:** Our goal of finding a *useful* balanced model was successful. We avoided the "100% recall" trap and built a model that provides actionable intelligence.

4.  **Hypothesis Confirmed:** The 8-feature model (without `max_price`) was highly effective, proving that `max_price` was likely noise.
