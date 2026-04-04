# E-commerce Cart Abandonment Prediction & User Segmentation

This project analyzes over 109 million user events from a large multi-category e-commerce store (October & November 2019) to achieve two main goals:

1. Predict cart abandonment using a balanced, production-ready machine learning model.
2. Segment customers into value- and behavior-based clusters for actionable marketing and product insights.

The pipeline is designed for high-performance computing (HPC) environments, prioritizing efficient CPU-based processing for large-scale aggregation and clustering, alongside model training using XGBoost and LightGBM.

---

## Project Goals

1. **Predict Abandonment**: Build a binary classifier to identify user sessions where a user adds at least one item to their cart but does not complete a purchase.
2. **Optimize for Balance**: Develop a model with a realistic balance of high recall (catching at-risk users) and high precision (minimizing noise during interventions, such as coupons or emails).
3. **User Segmentation**: Cluster users based on behavioral shopping styles, such as browsing vs. decisive buying, to enable targeted personalization.

---

## Project Structure

### Core Pipeline Scripts

| File | Description |
| :--- | :--- |
| **`process_data_safe.py`** | **Data Engineering:** Loads raw CSVs progressively to manage memory constraints. Cleans events, labels abandonment, engineers session features, and aggregates 109M+ events into ~23M sessions. Produces `dataset/sessions_df_final_project.parquet`. |
| **`run_comprehensive_bakeoff_resumable.py`** | **Model Training & Bake-Off:** Performs a stratified split and tests Logistic Regression, Random Forest, XGBoost, and LightGBM models. Uses `GridSearchCV` to tune top candidates. Logs progress to safely resume after interruptions. |
| **`train_and_save_champion.py`** | **Champion Model:** Retrains the tuned XGBoost model on the full training data, computes feature importance, and exports the model layout and plots. |
| **`clustering.py`** | **User Clustering:** Extracts behavioral features from sessions and applies K-Means clustering to distinguish between 8 behavioral patterns. |
| **`EDA.ipynb`** | An exploratory notebook utilized for initial dataset checks and formulation of hypotheses. |

### Results & Artifacts (`results/` folder)

| File / Pattern | Description |
| :--- | :--- |
| **`all_model_results.csv`** | Final leaderboard detailing evaluation metrics alongside training times. |
| **`running_results.csv`** | Intermediate logs captured by the resumable script. |
| **`champion_xgb_model.json`** | The finalized, deployable XGBoost model. |
| **`champion_feature_importance.png`** | Assessment of features driving predictions. |
| **`*_confusion_matrix.png`** | Evaluative plots for all tested models. |
| **`23m_event_cluster_analysis_detailed.csv`** | Analytics break-down specific to each behavioral cluster. |
| **`23m_event_clustering_results_detailed.png`** | A dashboard outlining cluster statistics and behaviors. |
| **`cluster_names_mapping.csv`** | A reference table mapping cluster IDs to descriptive names. |

---

## Setup & Installation

This project targets a Python 3.10+ environment and was designed for multi-cored environments.

### 1. Source the Data

This repository does not contain the raw data. Please download `2019-Oct.csv` and `2019-Nov.csv` from the [eCommerce Events History dataset on Kaggle](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store). Note that you should rename them to `2019-Oct-Optimized.parquet` and `2019-Nov-Optimized.parquet`, or use the original CSVs upon making slight adjustments to the `process_data_safe.py` script.

Place them under the `dataset/` directory.

### 2. Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate
# Windows: .\venv\Scripts\activate
```

### 3. Install Requirements

```bash
pip install pandas pyarrow scikit-learn xgboost lightgbm matplotlib seaborn jupyter
```

---

## Execution Instructions

Note: A full end-to-end run can take several hours depending on hardware specifications.

### 1. Data Processing

Aggregates the raw events into session-level mappings.

```bash
python3 process_data_safe.py
```

### 2. Model Training & Bake-Off

Executes the model comparison pipeline.

```bash
nohup python3 run_comprehensive_bakeoff_resumable.py > training.log 2>&1 &
```

### 3. Save the Champion Model

Creates the deployable model.

```bash
python3 train_and_save_champion.py
```

### 4. User Segmentation

Segments users logically through clustering.

```bash
python3 clustering.py
```

---

## Modeling Insights & Leaderboard

### Target Definition

Each session is labeled `1` for abandoned if an item was added to the cart but not purchased, and `0` otherwise.

### Feature Highlights

Key features include counts of views, carts, purchases, total duration, and tracking returning users. Contextual tests showed `max_price` was noisy, thus it was systematically excluded.

### Final Leaderboard Highlight

The **Tuned XGBoost** model achieved a weighted F1-score of **0.9697**, showcasing ~80% recall and ~71.5% precision. This performance maintains strong business viability and confidently flags truly at-risk sessions.

### Key Learnings

1. Default XGBoost performs similarly to tuned models, suggesting that feature engineering quality superseded hyperparameter tuning.
2. The number of cart additions (`num_carts`) is the dominant predictive feature (~96% importance).
3. The selected model mitigates pathological predictions while supplying actionable metrics for intervention systems.

---

## User Segmentation

Through unsupervised learning, we mapped behaviors across 8 distinct clusters based on metrics such as exploration depth and cart-to-view ratios. Identified segments typically fall into high-risk abandoners, loyal window shoppers, premium shoppers, and quick browsers. Check the generated visualizations to perceive qualitative shifts across the user base.

---

## License & Attribution

Data originates from the public eCommerce behavior dataset listed on Kaggle. The feature engineering and predictive frameworks built here are intended for educational and analytical progression, easily adaptable to internal domains given standard policy alignments. 

Thank you for exploring this project!
