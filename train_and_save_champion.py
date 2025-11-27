import pandas as pd
import numpy as np
import os
import gc
import time
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- 1. Configuration & Setup ---
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

DATA_FILE = "dataset/sessions_df_final_project.parquet"
RESULTS_DIR = "results"
MODEL_SAVE_PATH = os.path.join(RESULTS_DIR, "champion_xgb_model.json")
PLOT_SAVE_PATH = os.path.join(RESULTS_DIR, "champion_feature_importance.png")

def main():
    print("======================================================")
    print("  CHAMPION MODEL TRAINING & SAVING SCRIPT")
    print("======================================================")

    # --- 1. Load Data & Define Features ---
    print(f"\n--- 1. Loading {DATA_FILE} ---")
    df = pd.read_parquet(DATA_FILE)
    
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

    # --- 2. Train-Test Split (Must be identical to before) ---
    print("\n--- 3. Performing 70/30 Train-Test Split (with Stratify) ---")
    # We only need the training set to re-train the final model
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    del X, y
    gc.collect()

    # --- 3. Apply StandardScaler ---
    print("\n--- 4. Applying StandardScaler ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    del X_train # Free memory
    gc.collect()
    print("  -> Data prep complete.")

    # --- 4. Define and Train the Champion Model ---
    print("\n--- 5. Training the Champion Model (XGBoost_Tuned) ---")
    
    # These are the winning parameters from your log file
    champion_params = {
        'learning_rate': 0.05,
        'max_depth': 10,
        'n_estimators': 300,
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'logloss'
    }
    
    champion_model = xgb.XGBClassifier(**champion_params)
    
    start_time = time.time()
    champion_model.fit(X_train_scaled, y_train)
    elapsed = time.time() - start_time
    
    print(f"  -> Champion model trained in {elapsed:.2f} seconds.")

    # --- 5. Save the Trained Model ---
    print(f"\n--- 6. Saving model to {MODEL_SAVE_PATH} ---")
    champion_model.save_model(MODEL_SAVE_PATH)
    print("  -> Model saved successfully.")

    # --- 6. Generate and Save Feature Importance ---
    print(f"\n--- 7. Generating Feature Importance Plot ---")
    
    # Create a DataFrame for plotting
    importances = champion_model.feature_importances_
    feature_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print("Feature Importances:")
    print(feature_df)
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_df, palette='viridis')
    plt.title('Champion Model (XGBoost) Feature Importance')
    plt.xlabel('Importance Score (Gain)')
    plt.ylabel('Feature')
    
    # Save the plot
    plt.savefig(PLOT_SAVE_PATH, bbox_inches='tight')
    print(f"  -> Feature importance plot saved to {PLOT_SAVE_PATH}")
    
    print("\n======================================================")
    print("  ✅✅✅ SCRIPT COMPLETE ✅✅✅")
    print(f"  Champion model is saved and ready for use.")
    print("======================================================")

if __name__ == "__main__":
    main()
