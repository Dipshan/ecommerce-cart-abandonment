import pandas as pd
import numpy as np
import os
import gc
import time
import warnings

# --- 1. Configuration & Setup ---
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
pd.options.mode.chained_assignment = None  # Suppress SettingWithCopyWarning

OCT_PARQUET = "dataset/2019-Oct-Optimized.parquet"
NOV_PARQUET = "dataset/2019-Nov-Optimized.parquet"
FINAL_OUTPUT_FILE = "dataset/sessions_df_final_project.parquet"

def process_file_to_sessions(filepath):
    """
    Runs the entire midterm feature engineering pipeline
    on a *single* optimized parquet file.
    This is the memory-safe function.
    """
    print(f"\n--- Processing {filepath} ---")
    
    # 1. Load the entire Parquet file
    print("  Step 1: Loading event data...")
    start_load_time = time.time()
    try:
        df = pd.read_parquet(filepath)
        print(f"  -> Loaded {len(df)} events in {time.time() - start_load_time:.2f}s")
    except Exception as e:
        print(f"  ERROR loading {filepath}: {e}")
        return None

    # 2. Clean (price > 0)
    print("  Step 2: Cleaning data (price > 0)...")
    start_clean_time = time.time()
    len_before = len(df)
    df = df[df['price'] > 0].copy()
    len_after = len(df)
    print(f"  -> Removed {len_before - len_after} rows in {time.time() - start_clean_time:.2f}s")

    # 3. Create Labels (Your 'is_abandoned' logic)
    print("  Step 3: Creating 'is_abandoned' target label... (This is the long step)")
    start_label_time = time.time()
    df['has_cart'] = df.groupby('user_session', observed=True)['event_type'].transform(lambda x: 'cart' in x.values)
    df['has_purchase'] = df.groupby('user_session', observed=True)['event_type'].transform(lambda x: 'purchase' in x.values)
    df['is_abandoned'] = (df['has_cart'].astype(bool) & ~df['has_purchase'].astype(bool)).astype(int)
    df.drop(columns=['has_cart', 'has_purchase'], inplace=True)
    print(f"  -> Step 3 complete in {time.time() - start_label_time:.2f}s")
    
    # 4. Aggregate: Event Type Counts (Your pivot_table)
    print("  Step 4: Aggregating event counts (pivot)...")
    start_pivot_time = time.time()
    event_type_counts = pd.pivot_table(
        df, values='event_time', index='user_session', 
        columns='event_type', aggfunc='count', fill_value=0
    )
    event_type_counts.rename(columns={'view': 'num_views', 'cart': 'num_carts', 'purchase': 'num_purchases'}, inplace=True)
    print(f"  -> Step 4 complete in {time.time() - start_pivot_time:.2f}s")

    # 5. Aggregate: Other Features (Your groupby.agg)
    print("  Step 5: Aggregating all other features...")
    start_agg_time = time.time()
    sessions_df_enriched = df.groupby('user_session', observed=True).agg(
        session_start=('event_time', 'min'),
        session_end=('event_time', 'max'),
        products_viewed=('product_id', 'nunique'),
        avg_price=('price', 'mean'),
        max_price=('price', 'max'),
        brands_viewed=('brand', 'nunique'),
        categories_viewed=('category_code', 'nunique'),
        is_abandoned=('is_abandoned', 'first'),
        user_id=('user_id', 'first') # Added user_id
    )
    print(f"  -> Step 5 complete in {time.time() - start_agg_time:.2f}s")

    # 6. Combine Aggregations
    print("  Step 6: Combining features...")
    sessions_df_enriched = sessions_df_enriched.join(event_type_counts)

    # 7. Final Feature Creation
    print("  Step 7: Creating duration_seconds...")
    sessions_df_enriched['duration_seconds'] = (sessions_df_enriched['session_end'] - sessions_df_enriched['session_start']).dt.total_seconds()
    
    # Keep only the columns we need for the next step
    final_cols = [
        'session_start', 'user_id', 'duration_seconds', 'products_viewed',
        'avg_price', 'max_price', 'brands_viewed', 'categories_viewed',
        'num_carts', 'num_views', 'num_purchases', 'is_abandoned'
    ]
    final_cols_exist = [col for col in final_cols if col in sessions_df_enriched.columns]
    sessions_df_final = sessions_df_enriched[final_cols_exist]

    del df, event_type_counts, sessions_df_enriched
    gc.collect()
    
    print(f"-> Processing complete for {filepath}.")
    print(f"-> Returning session DataFrame with shape: {sessions_df_final.shape}")
    return sessions_df_final

def main():
    """
    Main function to run the memory-safe processing pipeline.
    """
    print("======================================================")
    print("  STARTING MEMORY-SAFE DATA PROCESSING PIPELINE")
    print("======================================================")
    
    # --- STAGE 1: PROCESS OCTOBER ---
    start_oct_time = time.time()
    sessions_oct = process_file_to_sessions(OCT_PARQUET)
    if sessions_oct is None:
        print("FATAL: October processing failed. Exiting.")
        return
    print(f"✅ October processing complete in {(time.time() - start_oct_time) / 60:.2f} minutes.")

    # --- STAGE 2: PROCESS NOVEMBER ---
    start_nov_time = time.time()
    sessions_nov = process_file_to_sessions(NOV_PARQUET)
    if sessions_nov is None:
        print("FATAL: November processing failed. Exiting.")
        return
    print(f"✅ November processing complete in {(time.time() - start_nov_time) / 60:.2f} minutes.")

    # --- STAGE 3: COMBINE SESSION-LEVEL DATA ---
    print("\n--- 3. COMBINING SESSION-LEVEL DATA ---")
    start_combine_time = time.time()
    
    sessions_df_final = pd.concat([sessions_oct, sessions_nov])
    print(f"  -> Combined DataFrame shape: {sessions_df_final.shape}")
    
    del sessions_oct, sessions_nov
    gc.collect()
    print(f"  -> Combine complete in {time.time() - start_combine_time:.2f}s")

    # --- STAGE 4: FINAL FEATURE ENGINEERING ---
    print("\n--- 4. ENGINEERING FINAL FEATURES ---")
    start_ff_time = time.time()
    
    # Create 'is_returning_user' (Hypothesis A)
    print("  4a. Engineering 'is_returning_user' (Hypothesis A)...")
    user_first_session_time = sessions_df_final.groupby('user_id')['session_start'].min().to_dict()
    sessions_df_final['user_first_session_time'] = sessions_df_final['user_id'].map(user_first_session_time)
    sessions_df_final['is_returning_user'] = (
        sessions_df_final['session_start'] > sessions_df_final['user_first_session_time']
    ).astype(int)
    
    print(f"  -> Final features created in {time.time() - start_ff_time:.2f}s")

    # --- STAGE 5: FINAL CLEANUP & SAVE ---
    print("\n--- 5. FINALIZING AND SAVING ---")
    
    features_to_keep = [
        'duration_seconds', 'products_viewed', 'avg_price', 'max_price',
        'brands_viewed', 'categories_viewed', 'num_carts', 'num_views',
        'is_returning_user', 'is_abandoned'
    ]
    
    sessions_df_model_ready = sessions_df_final[features_to_keep]
    
    is_abandoned_col = sessions_df_model_ready.pop('is_abandoned')
    sessions_df_model_ready['is_abandoned'] = is_abandoned_col
    
    print(f"  Final model-ready shape: {sessions_df_model_ready.shape}")

    print(f"  Saving final file to '{FINAL_OUTPUT_FILE}'...")
    start_save_time = time.time()
    sessions_df_model_ready.to_parquet(FINAL_OUTPUT_FILE, index=True)
    print(f"  -> Save complete in {time.time() - start_save_time:.2f}s")

    print("\n--- Final Class Imbalance Check ---")
    print(sessions_df_model_ready['is_abandoned'].value_counts(normalize=True))
    
    print("\n======================================================")
    print("  ✅✅✅ SCRIPT COMPLETE ✅✅✅")
    print(f"  Final file is ready at: {FINAL_OUTPUT_FILE}")
    print("======================================================")

if __name__ == "__main__":
    total_script_start = time.time()
    main()
    total_script_end = time.time()
    print(f"\nTotal script runtime: {(total_script_end - total_script_start) / 60:.2f} minutes")
