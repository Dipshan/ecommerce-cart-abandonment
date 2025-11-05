import pandas as pd
import numpy as np
import os
import gc
import time
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

# --- 1. Configuration & Setup ---
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
pd.options.mode.chained_assignment = None

DATA_FILE = "dataset/sessions_df_final_project.parquet"
RESULTS_DIR = "results" # Folder to save outputs
LOG_FILE = os.path.join(RESULTS_DIR, "running_results.csv")
FINAL_REPORT_FILE = os.path.join(RESULTS_DIR, "all_model_results.csv")

def save_results(model_name, y_test, y_pred, train_time, is_tuned=False):
    """
    Generates a classification report, saves a confusion matrix plot,
    and returns a dictionary of key metrics for the CSV.
    This function will also append the result to the running log file.
    """
    print(f"--- Evaluating {model_name} ---")
    
    # 1. Generate text report and metrics dictionary
    target_names = ['Not Abandoned (0)', 'Abandoned (1)']
    report_str = classification_report(y_test, y_pred, target_names=target_names)
    report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    
    print(report_str)
    
    # 2. Save Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    png_path = os.path.join(RESULTS_DIR, f"{model_name}_cm.png")
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names,
                yticklabels=target_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(png_path, bbox_inches='tight')
    plt.close() # Close the plot to save memory
    print(f"-> Saved confusion matrix to {png_path}")

    # 3. Prepare dictionary for CSV
    metrics = {
        'model_name': model_name,
        'is_tuned': is_tuned,
        'train_time_sec': round(train_time, 2),
        'accuracy': round(report_dict['accuracy'], 4),
        'abandoned_precision': round(report_dict['Abandoned (1)']['precision'], 4),
        'abandoned_recall': round(report_dict['Abandoned (1)']['recall'], 4),
        'abandoned_f1': round(report_dict['Abandoned (1)']['f1-score'], 4),
        'not_abandoned_precision': round(report_dict['Not Abandoned (0)']['precision'], 4),
        'not_abandoned_recall': round(report_dict['Not Abandoned (0)']['recall'], 4),
        'not_abandoned_f1': round(report_dict['Not Abandoned (0)']['f1-score'], 4),
        'weighted_avg_precision': round(report_dict['weighted avg']['precision'], 4),
        'weighted_avg_recall': round(report_dict['weighted avg']['recall'], 4),
        'weighted_avg_f1': round(report_dict['weighted avg']['f1-score'], 4)
    }
    
    # 4. Append result to the running log file
    try:
        if os.path.exists(LOG_FILE):
            # Append without header
            pd.DataFrame([metrics]).to_csv(LOG_FILE, mode='a', header=False, index=False)
        else:
            # Create new file with header
            pd.DataFrame([metrics]).to_csv(LOG_FILE, mode='w', header=True, index=False)
        print(f"-> Results for {model_name} appended to {LOG_FILE}")
    except Exception as e:
        print(f"Error saving to log file: {e}")
        
    return metrics

def main():
    """
    Main function to run the full comprehensive modeling pipeline.
    This script is resumable.
    """
    print("======================================================")
    print("  STARTING COMPREHENSIVE & RESUMABLE MODELING PIPELINE")
    print("  (No scale_pos_weight, No max_price, GridSearch on Top 2)")
    print(f"  Using {os.cpu_count()} CPU cores.")
    print("======================================================")

    # List to hold all our result dictionaries
    all_results = []
    
    # Dictionary to rank models for tuning
    model_scores = {}
    
    # --- 0. Resumability Check ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Created/verified results directory at: {RESULTS_DIR}")
    
    models_already_run = []
    if os.path.exists(LOG_FILE):
        print(f"Found existing log file: {LOG_FILE}")
        try:
            df_log = pd.read_csv(LOG_FILE)
            models_already_run = df_log['model_name'].tolist()
            all_results = df_log.to_dict('records')
            # Pre-populate scores for Top 2 selection
            for record in all_results:
                if not record['is_tuned']:
                    model_scores[record['model_name']] = record['weighted_avg_f1']
            print(f"Loaded results for: {models_already_run}")
        except Exception as e:
            print(f"Warning: Could not read log file. Starting from scratch. Error: {e}")
            models_already_run = []
            all_results = []

    # --- 1. Load Data, Split, and Scale (This always needs to run) ---
    print(f"\n--- 1. Loading {DATA_FILE} ---")
    df = pd.read_parquet(DATA_FILE)
    print(f"  -> Loaded {DATA_FILE}. Shape: {df.shape}")

    print("\n--- 2. Defining 8-Feature Set (No max_price) ---")
    y = df['is_abandoned']
    features = [
        'duration_seconds', 'products_viewed', 'avg_price',
        'brands_viewed', 'categories_viewed', 'num_carts', 'num_views',
        'is_returning_user'
    ]
    X = df[features]
    del df
    gc.collect()

    print("\n--- 3. Performing 70/30 Train-Test Split (with Stratify) ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    del X, y
    gc.collect()

    print("\n--- 4. Applying StandardScaler ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("  -> Data prep complete.")

    # --- STAGE 1: DEFAULT MODEL BAKE-OFF ---
    print("\n\n======================================================")
    print("  STAGE 1: DEFAULT PARAMETER BAKE-OFF")
    print("======================================================")

    models_to_run = [
        ('LogisticRegression', LogisticRegression(random_state=42, n_jobs=-1, max_iter=1000)),
        ('RandomForest', RandomForestClassifier(random_state=42, n_jobs=-1)),
        ('XGBoost', xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')),
        ('LightGBM', lgb.LGBMClassifier(random_state=42, n_jobs=-1))
    ]

    for model_name, model in models_to_run:
        if model_name in models_already_run:
            print(f"\nSKIPPING: {model_name} (Default) - Results already in log.")
            continue # Skip to the next model
            
        print(f"\n--- Training {model_name} (Default) ---")
        start_time = time.time()
        
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        train_time = time.time() - start_time
        print(f"-> {model_name} trained in {train_time:.2f}s")
        
        metrics = save_results(model_name, y_test, y_pred, train_time, is_tuned=False)
        all_results.append(metrics)
        model_scores[model_name] = metrics['weighted_avg_f1']

    # --- STAGE 2: IDENTIFY TOP 2 MODELS ---
    print("\n\n======================================================")
    print("  STAGE 2: IDENTIFYING TOP 2 MODELS (by Weighted F1)")
    print("======================================================")
    
    if not model_scores:
         print("ERROR: No model scores found. Cannot determine top 2.")
         return

    sorted_models = sorted(model_scores.items(), key=lambda item: item[1], reverse=True)
    top_2_names = [sorted_models[0][0], sorted_models[1][0]]
    
    print(f"Default Model Rankings (by Weighted F1):")
    for i, (name, score) in enumerate(sorted_models):
        print(f"  {i+1}. {name}: {score:.4f}")
    print(f"\nSelected for tuning: {top_2_names[0]} and {top_2_names[1]}")


    # --- STAGE 3: GRIDSEARCHCV FOR TOP 2 ---
    print("\n\n======================================================")
    print("  STAGE 3: HYPERPARAMETER TUNING (GridSearchCV)")
    print("======================================================")

    param_grids = {
        'LogisticRegression': {'C': [0.1, 1, 10], 'class_weight': [None, 'balanced']},
        'RandomForest': {'n_estimators': [100, 200], 'max_depth': [10, 30, None], 'min_samples_leaf': [1, 5], 'class_weight': [None, 'balanced']},
        'XGBoost': {'n_estimators': [100, 200, 300], 'learning_rate': [0.05, 0.1], 'max_depth': [5, 10, 15]},
        'LightGBM': {'n_estimators': [100, 200, 300], 'learning_rate': [0.05, 0.1], 'num_leaves': [31, 50, 70], 'is_unbalance': [True, False]}
    }
    
    base_estimators = {
        'LogisticRegression': LogisticRegression(random_state=42, n_jobs=-1, max_iter=1000),
        'RandomForest': RandomForestClassifier(random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss'),
        'LightGBM': lgb.LGBMClassifier(random_state=42, n_jobs=-1)
    }

    for model_name in top_2_names:
        tuned_model_name = f"{model_name}_Tuned"
        
        if tuned_model_name in models_already_run:
            print(f"\nSKIPPING: {tuned_model_name} - Results already in log.")
            continue # Skip to the next model
            
        print(f"\n--- Tuning {model_name} (GridSearchCV) ---")
        
        n_jobs_setting = -1
        if model_name == 'LightGBM':
            n_jobs_setting = 8 # MEMORY-SAFE FIX
            print(f"*** Applying MEMORY-SAFE FIX: Setting n_jobs=8 for LightGBM ***")
        
        print(f"Parameter grid: {param_grids[model_name]}")
        
        grid_search = GridSearchCV(
            estimator=base_estimators[model_name],
            param_grid=param_grids[model_name],
            scoring='f1_weighted',
            cv=3,
            verbose=3,
            n_jobs=n_jobs_setting
        )
        
        start_grid_time = time.time()
        grid_search.fit(X_train_scaled, y_train)
        grid_train_time = time.time() - start_grid_time
        
        print(f"\n-> GridSearchCV for {model_name} finished in {grid_train_time / 60:.2f} minutes.")
        
        best_model = grid_search.best_estimator_
        print(f"Best Parameters for {model_name}:")
        print(grid_search.best_params_)
        
        y_pred_tuned = best_model.predict(X_test_scaled)
        
        # Save results and append to log
        metrics_tuned = save_results(tuned_model_name, y_test, y_pred_tuned, grid_train_time, is_tuned=True)
        all_results.append(metrics_tuned)

    # --- STAGE 4: SAVE FINAL CSV REPORT ---
    print("\n\n======================================================")
    print("  STAGE 4: SAVING FINAL CSV REPORT")
    print("======================================================")
    
    # Load the definitive log file, which has all results
    try:
        results_df = pd.read_csv(LOG_FILE)
        
        # Sort by our key metric to see the final winner
        results_df.sort_values(by='weighted_avg_f1', ascending=False, inplace=True)
        
        results_df.to_csv(FINAL_REPORT_FILE, index=False)
        
        print(f"✅ All model results saved to {FINAL_REPORT_FILE}")
        print("\nFinal Model Leaderboard (by Weighted F1):")
        print(results_df[['model_name', 'weighted_avg_f1', 'abandoned_recall', 'abandoned_precision', 'train_time_sec']])
    
    except FileNotFoundError:
        print("ERROR: Log file not found. Cannot generate final report.")
    except Exception as e:
        print(f"Error generating final report: {e}")

    
    print("\n======================================================")
    print("  ✅✅✅ COMPREHENSIVE BAKE-OFF COMPLETE ✅✅✅")
    print(f"  All reports and plots saved to: {RESULTS_DIR}")
    print("======================================================")

if __name__ == "__main__":
    total_script_start = time.time()
    main()
    total_script_end = time.time()
    print(f"\nTotal script runtime: {(total_script_end - total_script_start) / 60:.2f} minutes")
