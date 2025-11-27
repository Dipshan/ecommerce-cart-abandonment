import os
import time
import warnings
import gc
import pickle

# CPU Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# GPU Libraries (Import safely)
try:
    import cudf
    import cupy as cp
    print("✅ RAPIDS (cudf) detected. Using GPU for data aggregation.")
except ImportError:
    print("⚠️ RAPIDS not found. This script requires a GPU environment.")
    exit(1)

# --- Configuration ---
warnings.filterwarnings('ignore')
OCT_FILE = "dataset/2019-Oct-Optimized.parquet"
NOV_FILE = "dataset/2019-Nov-Optimized.parquet"
RESULTS_DIR = "results"
INTERMEDIATE_FILE = os.path.join(RESULTS_DIR, "user_level_data.parquet")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================================
# PART 1: GPU DATA AGGREGATION (The Fast Part)
# ==========================================
def process_month_gpu(filepath):
    """Loads a month file into GPU memory and aggregates it."""
    print(f"--- Processing {filepath} on GPU ---")
    start = time.time()
    
    cols = ['user_id', 'user_session', 'event_type', 'price', 'event_time']
    gdf = cudf.read_parquet(filepath, columns=cols)
    
    # 1. Filter Purchases
    purchases = gdf[gdf['event_type'] == 'purchase']
    user_spend = purchases.groupby('user_id')['price'].sum()
    user_spend.name = 'total_spend'
    
    user_orders = purchases.groupby('user_id')['price'].count()
    user_orders.name = 'total_orders'
    
    # 2. Session counts & Last Active
    user_sessions = gdf.groupby('user_id')['user_session'].nunique()
    user_sessions.name = 'total_sessions'
    
    last_active = gdf.groupby('user_id')['event_time'].max()
    last_active.name = 'last_active'
    
    # 3. Behavioral: Views and Carts (Manual pivot for speed)
    gdf['is_view'] = (gdf['event_type'] == 'view').astype('int8')
    gdf['is_cart'] = (gdf['event_type'] == 'cart').astype('int8')
    
    event_counts = gdf.groupby('user_id')[['is_view', 'is_cart']].sum()
    event_counts.rename(columns={'is_view': 'total_views', 'is_cart': 'total_carts'}, inplace=True)
    
    # --- Combine Stats ---
    user_stats = event_counts
    user_stats['total_spend'] = user_spend
    user_stats['total_orders'] = user_orders
    user_stats['total_sessions'] = user_sessions
    user_stats['last_active'] = last_active
    
    user_stats = user_stats.fillna(0)
    print(f"-> Processed {len(user_stats)} users in {time.time() - start:.2f}s")
    
    del gdf, purchases, user_spend, user_orders, user_sessions, last_active
    gc.collect()
    return user_stats

def run_gpu_aggregation():
    print("\n[PHASE 1] Building User-Level Dataset on GPU...")
    
    oct_stats = process_month_gpu(OCT_FILE)
    nov_stats = process_month_gpu(NOV_FILE)
    
    print("Combining months on GPU...")
    combined = cudf.concat([oct_stats, nov_stats])
    
    # Group by index (user_id) to merge duplicate users across months
    combined['user_id'] = combined.index
    user_gdf = combined.groupby('user_id').agg({
        'total_spend': 'sum',
        'total_orders': 'sum',
        'total_sessions': 'sum',
        'last_active': 'max',
        'total_views': 'sum',
        'total_carts': 'sum'
    })
    
    print(f"-> Aggregated into {len(user_gdf)} unique users.")
    
    # Feature Engineering on GPU
    print("[PHASE 2] Engineering Features on GPU...")
    snapshot_date = user_gdf['last_active'].max()
    user_gdf['Recency'] = (snapshot_date - user_gdf['last_active']).dt.days
    user_gdf['Frequency'] = user_gdf['total_sessions']
    user_gdf['Monetary'] = user_gdf['total_spend']
    
    user_gdf['View_to_Cart'] = user_gdf['total_carts'] / user_gdf['total_views'].replace(0, 1)
    user_gdf['Cart_Conversion'] = user_gdf['total_orders'] / user_gdf['total_carts'].replace(0, 1)
    user_gdf['Views_per_Session'] = user_gdf['total_views'] / user_gdf['total_sessions']
    
    # Save to Parquet (Robust hand-off to CPU)
    # We convert to Pandas first to ensure 100% safe I/O for the next step
    print(f"Saving intermediate file to {INTERMEDIATE_FILE}...")
    user_gdf.to_pandas().to_parquet(INTERMEDIATE_FILE)
    print("-> GPU Work Complete.")
    
    del oct_stats, nov_stats, combined, user_gdf
    gc.collect()

# ==========================================
# PART 2: CPU CLUSTERING & ANALYSIS (The Safe Part)
# ==========================================
def run_cpu_clustering():
    print("\n[PHASE 3] Loading Data for CPU Clustering...")
    df = pd.read_parquet(INTERMEDIATE_FILE)
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    print(f"-> Loaded {len(df)} users.")

    # --- RFM CLUSTERING ---
    print("\n--- Performing RFM Clustering ---")
    rfm_cols = ['Recency', 'Frequency', 'Monetary']
    rfm_data = df[rfm_cols].copy()

    # Log Transform & Cap Outliers
    for col in rfm_cols:
        cap = rfm_data[col].quantile(0.99)
        rfm_data[col] = np.where(rfm_data[col] > cap, cap, rfm_data[col])
        if col != 'Recency':
            rfm_data[col] = np.log1p(rfm_data[col])

    scaler_rfm = StandardScaler()
    rfm_scaled = scaler_rfm.fit_transform(rfm_data)

    kmeans_rfm = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['RFM_Cluster'] = kmeans_rfm.fit_predict(rfm_scaled)
    
    # Save RFM Model
    pickle.dump(kmeans_rfm, open(os.path.join(RESULTS_DIR, "kmeans_rfm_model.pkl"), "wb"))

    # --- STYLE CLUSTERING ---
    print("\n--- Performing Shopping Style Clustering ---")
    style_cols = ['View_to_Cart', 'Cart_Conversion', 'Views_per_Session']
    style_data = df[style_cols].copy()

    for col in style_cols:
        cap = style_data[col].quantile(0.99)
        style_data[col] = np.where(style_data[col] > cap, cap, style_data[col])

    scaler_style = StandardScaler()
    style_scaled = scaler_style.fit_transform(style_data)

    kmeans_style = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['Style_Cluster'] = kmeans_style.fit_predict(style_scaled)
    
    # Save Style Model
    pickle.dump(kmeans_style, open(os.path.join(RESULTS_DIR, "kmeans_style_model.pkl"), "wb"))

    # --- SAVING RESULTS ---
    print("\n[PHASE 4] Saving Profiles & Plots...")

    # 1. Profiles
    rfm_profile = df.groupby('RFM_Cluster')[rfm_cols].mean()
    style_profile = df.groupby('Style_Cluster')[style_cols].mean()

    # Print Inspection Results (Replaces inspect_clusters.py)
    print("\n>>> FINAL RFM PROFILE (Use to name clusters):")
    print(rfm_profile.round(2))
    rfm_profile.to_csv(os.path.join(RESULTS_DIR, "cluster_profile_rfm.csv"))

    print("\n>>> FINAL STYLE PROFILE (Use to name clusters):")
    print(style_profile.round(4))
    style_profile.to_csv(os.path.join(RESULTS_DIR, "cluster_profile_style.csv"))

    # 2. Plots (Sampled)
    plot_sample = df.sample(min(100000, len(df)), random_state=42)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=plot_sample, x='Frequency', y='Monetary', hue='RFM_Cluster', palette='viridis', alpha=0.6)
    plt.title('User Segments: Frequency vs Monetary (RFM)')
    plt.yscale('log'); plt.xscale('log')
    plt.savefig(os.path.join(RESULTS_DIR, "rfm_clusters.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=plot_sample, x='Views_per_Session', y='View_to_Cart', hue='Style_Cluster', palette='coolwarm', alpha=0.6)
    plt.title('Shopping Styles')
    plt.savefig(os.path.join(RESULTS_DIR, "style_clusters.png"))
    plt.close()

    print(f"\n✅ SUCCESS. All files saved to {RESULTS_DIR}")

def main():
    # Step 1: Use GPU to crunch the massive files
    run_gpu_aggregation()
    
    # Step 2: Use CPU to do the clustering/plotting (Safe & Clean)
    run_cpu_clustering()

if __name__ == "__main__":
    main()
