# complete_23m_event_clustering_FIXED.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import gc
from datetime import datetime

# Filter warnings for cleaner output
warnings.filterwarnings('ignore')

class MassiveEventClustering:
    def __init__(self, oct_file, nov_file):
        self.oct_file = oct_file
        self.nov_file = nov_file
        self.raw_events = None
        self.session_features = None
        
        # Ensure results directory exists
        if not os.path.exists('results'):
            os.makedirs('results')
        
    def load_and_combine_raw_events(self):
        """Load all 23M+ raw events from both months"""
        print("📥 LOADING 23M+ RAW EVENTS...")
        
        # Load October events
        print(f"  Loading October data from {self.oct_file}...")
        try:
            oct_events = pd.read_parquet(self.oct_file)
            print(f"  October: {len(oct_events):,} events")
        except FileNotFoundError:
            print(f"  ❌ ERROR: Could not find {self.oct_file}")
            raise

        # Load November events  
        print(f"  Loading November data from {self.nov_file}...")
        try:
            nov_events = pd.read_parquet(self.nov_file)
            print(f"  November: {len(nov_events):,} events")
        except FileNotFoundError:
            print(f"  ❌ ERROR: Could not find {self.nov_file}")
            raise
        
        # Combine all events
        self.raw_events = pd.concat([oct_events, nov_events], ignore_index=True)
        print(f" COMBINED: {len(self.raw_events):,} TOTAL EVENTS")
        
        # Clean up memory
        del oct_events, nov_events
        gc.collect()
        
    def create_session_features_from_events(self):
        """Process 23M events into session-level features"""
        print("\n🔨 PROCESSING 23M EVENTS INTO SESSION FEATURES...")
        
        # Filter valid events (price > 0)
        valid_events = self.raw_events[self.raw_events['price'] > 0].copy()
        print(f"  Valid events (price > 0): {len(valid_events):,}")
        
        # Process in chunks to manage memory
        chunk_size = 5000000  # 5M events per chunk
        session_chunks = []
        
        for i in range(0, len(valid_events), chunk_size):
            chunk_end = min(i + chunk_size, len(valid_events))
            print(f"  Processing chunk: {i:,} to {chunk_end:,} events")
            
            chunk = valid_events.iloc[i:chunk_end].copy()
            
            # Create session features for this chunk
            session_features_chunk = self._process_event_chunk(chunk)
            session_chunks.append(session_features_chunk)
            
            # Clean memory
            del chunk
            gc.collect()
        
        # Combine all session chunks
        self.session_features = pd.concat(session_chunks, ignore_index=True)
        
        # Aggregate duplicate sessions (if any session spans chunks)
        self.session_features = self.session_features.groupby('user_session').agg({
            'session_start': 'min',
            'session_end': 'max', 
            'user_id': 'first',
            'total_events': 'sum',
            'view_count': 'sum',
            'cart_count': 'sum',
            'purchase_count': 'sum',
            'unique_products': 'sum',
            'unique_brands': 'sum',
            'unique_categories': 'sum',
            'avg_price': 'mean',
            'max_price': 'max',
            'total_price': 'sum',
            'is_abandoned': 'max'
        }).reset_index()
        
        print(f"✅ CREATED {len(self.session_features):,} SESSIONS FROM 23M EVENTS")
        
    def _process_event_chunk(self, event_chunk):
        """Process a chunk of events into session features"""
        
        # Event type counts per session
        event_counts = event_chunk.groupby(['user_session', 'event_type']).size().unstack(fill_value=0)
        
        # Ensure all columns exist
        for col in ['view', 'cart', 'purchase']:
            if col not in event_counts.columns:
                event_counts[col] = 0
                
        event_counts = event_counts.rename(columns={
            'view': 'view_count',
            'cart': 'cart_count', 
            'purchase': 'purchase_count'
        })
        
        # Session time features
        session_times = event_chunk.groupby('user_session').agg({
            'event_time': ['min', 'max'],
            'user_id': 'first'
        })
        session_times.columns = ['session_start', 'session_end', 'user_id']
        
        # Product and category features
        product_features = event_chunk.groupby('user_session').agg({
            'product_id': 'nunique',
            'brand': 'nunique',
            'category_code': 'nunique',
            'price': ['mean', 'max', 'sum']
        })
        product_features.columns = [
            'unique_products', 'unique_brands', 'unique_categories',
            'avg_price', 'max_price', 'total_price'
        ]
        
        # Total events per session
        total_events = event_chunk.groupby('user_session').size().reset_index(name='total_events')
        
        # Combine all features
        session_features = session_times.join(event_counts, how='left')
        session_features = session_features.join(product_features, how='left')
        session_features = session_features.merge(total_events, on='user_session', how='left')
        
        # Fill NaN values for sessions missing certain event types
        for col in ['view_count', 'cart_count', 'purchase_count']:
            session_features[col] = session_features[col].fillna(0)
        
        # Create abandonment label
        session_features['is_abandoned'] = (
            (session_features['cart_count'] > 0) & 
            (session_features['purchase_count'] == 0)
        ).astype(int)
        
        return session_features.reset_index()
    
    def engineer_clustering_features(self):
        """Create features suitable for clustering"""
        print("\n Engineering Clustering Features...")
        
        # Basic session features
        self.session_features['duration_seconds'] = (
            self.session_features['session_end'] - self.session_features['session_start']
        ).dt.total_seconds()
        
        # Engagement intensity features
        self.session_features['events_per_minute'] = (
            self.session_features['total_events'] / (self.session_features['duration_seconds'] / 60)
        ).replace([np.inf, -np.inf], 0).fillna(0)
        
        self.session_features['cart_to_view_ratio'] = (
            self.session_features['cart_count'] / (self.session_features['view_count'] + 1)
        )
        
        self.session_features['purchase_to_cart_ratio'] = (
            self.session_features['purchase_count'] / (self.session_features['cart_count'] + 1)
        )
        
        # Exploration depth features
        self.session_features['products_per_event'] = (
            self.session_features['unique_products'] / (self.session_features['total_events'] + 1)
        )
        
        print(f"✅ ENGINEERED {self.session_features.shape[1]} FEATURES FOR CLUSTERING")
        
    def perform_clustering(self, n_clusters=8):
        """Perform K-means clustering on session features"""
        print(f"\n🎪 PERFORMING CLUSTERING ON {len(self.session_features):,} SESSIONS...")
        
        # Select features for clustering
        clustering_features = [
            'duration_seconds', 'total_events', 'view_count', 'cart_count',
            'unique_products', 'unique_brands', 'unique_categories', 
            'avg_price', 'events_per_minute', 'cart_to_view_ratio',
            'products_per_event', 'is_abandoned'
        ]
        
        X = self.session_features[clustering_features].copy()
        
        # Handle infinite and missing values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # Log transform skewed features
        skewed_features = ['duration_seconds', 'total_events', 'view_count', 'avg_price']
        for feature in skewed_features:
            if feature in X.columns:
                X[feature] = np.log1p(X[feature])
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform K-means clustering
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
            max_iter=100,
            verbose=1
        )
        
        self.session_features['behavior_cluster'] = kmeans.fit_predict(X_scaled)
        
        print(f"✅ CLUSTERING COMPLETE - {n_clusters} CLUSTERS CREATED")
        
        return kmeans, X_scaled
    
    def analyze_clusters(self):
        """Analyze and describe the clusters"""
        print("\n📊 ANALYZING CLUSTER BEHAVIORS...")
        
        cluster_analysis = self.session_features.groupby('behavior_cluster').agg({
            'duration_seconds': ['mean', 'std'],
            'total_events': 'mean',
            'view_count': 'mean',
            'cart_count': 'mean', 
            'purchase_count': 'mean',
            'unique_products': 'mean',
            'avg_price': 'mean',
            'is_abandoned': 'mean',
            'user_session': 'count'
        }).round(2)
        
        cluster_analysis.columns = [
            'avg_duration', 'std_duration', 'avg_events', 'avg_views',
            'avg_carts', 'avg_purchases', 'avg_products', 'avg_price',
            'abandonment_rate', 'session_count'
        ]
        
        cluster_analysis['percent_of_total'] = (
            cluster_analysis['session_count'] / len(self.session_features) * 100
        ).round(1)
        
        print("📈 CLUSTER SUMMARY:")
        print(cluster_analysis)
        
        return cluster_analysis

    def create_intelligent_cluster_names(self, cluster_analysis):
        """Create meaningful names for each cluster based on behavior patterns"""
        print("\n🏷️ CREATING INTELLIGENT CLUSTER NAMES...")
        
        cluster_names = {}
        cluster_descriptions = {}
        
        # Calculate percentile thresholds for each metric
        duration_33 = cluster_analysis['avg_duration'].quantile(0.33)
        duration_66 = cluster_analysis['avg_duration'].quantile(0.66)
        
        products_33 = cluster_analysis['avg_products'].quantile(0.33)
        products_66 = cluster_analysis['avg_products'].quantile(0.66)
        
        price_33 = cluster_analysis['avg_price'].quantile(0.33)
        price_66 = cluster_analysis['avg_price'].quantile(0.66)
        
        abandon_33 = cluster_analysis['abandonment_rate'].quantile(0.33)
        abandon_66 = cluster_analysis['abandonment_rate'].quantile(0.66)
        
        for cluster_id, row in cluster_analysis.iterrows():
            name_parts = []
            desc_parts = []
            
            # 1. Abandonment Risk (Primary)
            if row['abandonment_rate'] > abandon_66:
                name_parts.append("High-Risk")
                desc_parts.append(f"very high abandonment ({row['abandonment_rate']:.1%})")
            elif row['abandonment_rate'] > abandon_33:
                name_parts.append("Medium-Risk")
                desc_parts.append(f"moderate abandonment ({row['abandonment_rate']:.1%})")
            else:
                name_parts.append("Low-Risk")
                desc_parts.append(f"low abandonment ({row['abandonment_rate']:.1%})")
            
            # 2. Session Duration
            if row['avg_duration'] > duration_66:
                name_parts.append("Patient")
                desc_parts.append(f"long sessions ({row['avg_duration']:.0f}s)")
            elif row['avg_duration'] > duration_33:
                name_parts.append("Balanced")
                desc_parts.append(f"medium sessions ({row['avg_duration']:.0f}s)")
            else:
                name_parts.append("Quick")
                desc_parts.append(f"short sessions ({row['avg_duration']:.0f}s)")
            
            # 3. Product Exploration
            if row['avg_products'] > products_66:
                name_parts.append("Explorer")
                desc_parts.append(f"views {row['avg_products']:.0f} products")
            elif row['avg_products'] > products_33:
                name_parts.append("Browser")
                desc_parts.append(f"views {row['avg_products']:.0f} products")
            else:
                name_parts.append("Shopper")
                desc_parts.append(f"views {row['avg_products']:.0f} products")
            
            # 4. Price Sensitivity
            if row['avg_price'] > price_66:
                name_parts.append("Premium")
                desc_parts.append(f"premium price (${row['avg_price']:.0f})")
            elif row['avg_price'] > price_33:
                name_parts.append("Mid-Range")
                desc_parts.append(f"mid-range (${row['avg_price']:.0f})")
            else:
                name_parts.append("Budget")
                desc_parts.append(f"low price (${row['avg_price']:.0f})")
            
            cluster_names[cluster_id] = " | ".join(name_parts)
            cluster_descriptions[cluster_id] = ", ".join(desc_parts)
            
            print(f"  Cluster {cluster_id}: {cluster_names[cluster_id]}")
            print(f"    Description: {cluster_descriptions[cluster_id]}")
        
        return cluster_names, cluster_descriptions

    def visualize_clusters(self, cluster_analysis, cluster_names, cluster_descriptions):
        """Create comprehensive cluster visualizations"""
        print("\n🎨 CREATING CLUSTER VISUALIZATIONS...")
        
        plt.figure(figsize=(20, 12))
        
        # Create named stats for plotting
        named_stats = cluster_analysis.copy()
        named_stats["cluster_name"] = named_stats.index.map(cluster_names)
        
        # =============================
        # PLOT 1 - Cluster Sizes (Pie Chart)
        # =============================
        plt.subplot(2, 3, 1)
        plt.pie(named_stats['session_count'], 
                labels=[f'Cluster {i}' for i in named_stats.index],
                autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
        plt.title('Cluster Size Distribution')
        
        # =============================
        # PLOT 2 - Abandonment Rates
        # =============================
        plt.subplot(2, 3, 2)
        sorted_stats = named_stats.sort_values('abandonment_rate', ascending=False)
        bars = plt.bar(range(len(sorted_stats)), sorted_stats['abandonment_rate'],
                      color=['#ff6b6b' if x > named_stats['abandonment_rate'].mean() 
                           else '#51cf66' for x in sorted_stats['abandonment_rate']])
        plt.xticks(range(len(sorted_stats)), [f'C{i}' for i in sorted_stats.index])
        plt.ylabel('Abandonment Rate')
        plt.title('Abandonment Rate by Cluster\n(Red = High Risk)')
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.1%}', ha='center', va='bottom', fontsize=9)
        
        # =============================
        # PLOT 3 - Duration vs Products
        # =============================
        plt.subplot(2, 3, 3)
        scatter = plt.scatter(named_stats['avg_duration'], 
                             named_stats['avg_products'],
                             s=named_stats['session_count']/named_stats['session_count'].max()*1000,
                             c=named_stats.index, cmap='viridis', alpha=0.7)
        
        for idx, row in named_stats.iterrows():
            plt.annotate(f'C{idx}', (row['avg_duration'], row['avg_products']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        plt.xlabel('Average Duration (seconds)')
        plt.ylabel('Average Products Viewed')
        plt.title('Session Duration vs Product Exploration\n(Size = Cluster Size)')
        plt.colorbar(scatter, label='Cluster ID')
        
        # =============================
        # PLOT 4 - Price vs Abandonment
        # =============================
        plt.subplot(2, 3, 4)
        plt.scatter(named_stats['avg_price'], 
                   named_stats['abandonment_rate'],
                   s=named_stats['session_count']/named_stats['session_count'].max()*1000,
                   c=named_stats.index, cmap='Set3', alpha=0.7)
        
        for idx, row in named_stats.iterrows():
            plt.annotate(f'C{idx}', (row['avg_price'], row['abandonment_rate']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        plt.xlabel('Average Price ($)')
        plt.ylabel('Abandonment Rate')
        plt.title('Price Sensitivity vs Abandonment')
        
        # =============================
        # PLOT 5 - Event Intensity
        # =============================
        plt.subplot(2, 3, 5)
        x_pos = np.arange(len(named_stats))
        width = 0.25
        
        plt.bar(x_pos - width, named_stats['avg_views'], width, label='Views', alpha=0.7)
        plt.bar(x_pos, named_stats['avg_carts'], width, label='Carts', alpha=0.7)
        plt.bar(x_pos + width, named_stats['avg_purchases'], width, label='Purchases', alpha=0.7)
        
        plt.xticks(x_pos, [f'C{i}' for i in named_stats.index])
        plt.ylabel('Average Count')
        plt.title('Event Intensity by Cluster')
        plt.legend()
        
        # =============================
        # PLOT 6 - Cluster Descriptions
        # =============================
        plt.subplot(2, 3, 6)
        plt.axis('off')
        
        summary_text = "CLUSTER SUMMARY:\n\n"
        for idx, row in named_stats.iterrows():
            summary_text += f"C{idx}: {cluster_names[idx]}\n"
            summary_text += f" - {row['percent_of_total']}% of sessions\n"
            summary_text += f" - {cluster_descriptions[idx][:40]}...\n\n"
        
        plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, 
                fontsize=9, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        output_path = 'results/23m_event_clustering_results_detailed.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ VISUALIZATIONS SAVED AS '{output_path}'")

    def save_results(self, cluster_names, cluster_descriptions):
        """Save all results with cluster names"""
        print("\n💾 SAVING RESULTS TO 'results/' DIRECTORY...")
        
        # Add cluster names to session features
        self.session_features['cluster_name'] = self.session_features['behavior_cluster'].map(cluster_names)
        self.session_features['cluster_description'] = self.session_features['behavior_cluster'].map(cluster_descriptions)
        
        # Save session features with clusters
        self.session_features.to_parquet('results/23m_event_session_clusters_with_names.parquet', index=False)
        
        # Save cluster analysis with names
        cluster_summary = self.session_features.groupby('behavior_cluster').agg({
            'duration_seconds': ['mean', 'std'],
            'total_events': 'mean',
            'view_count': 'mean',
            'cart_count': 'mean',
            'purchase_count': 'mean',
            'unique_products': 'mean',
            'avg_price': 'mean',
            'is_abandoned': 'mean',
            'user_session': 'count',
            'cluster_name': 'first',
            'cluster_description': 'first'
        })
        
        cluster_summary.to_csv('results/23m_event_cluster_analysis_detailed.csv')
        
        # Save cluster names mapping
        name_mapping = pd.DataFrame({
            'cluster_id': list(cluster_names.keys()),
            'cluster_name': list(cluster_names.values()),
            'cluster_description': list(cluster_descriptions.values())
        })
        name_mapping.to_csv('results/cluster_names_mapping.csv', index=False)
        
        print("✅ RESULTS SAVED SUCCESSFULLY")

def main():
    """Main execution function"""
    print("=" * 60)
    print("🚀 23 MILLION EVENT SESSION CLUSTERING PIPELINE")
    print("=" * 60)
    
    # Initialize the clustering engine with CORRECT PATHS
    cluster_engine = MassiveEventClustering(
        oct_file="dataset/2019-Oct-Optimized.parquet",
        nov_file="dataset/2019-Nov-Optimized.parquet"
    )
    
    # Step 1: Load all 23M+ events
    cluster_engine.load_and_combine_raw_events()
    
    # Step 2: Process events into session features
    cluster_engine.create_session_features_from_events()
    
    # Step 3: Engineer clustering features
    cluster_engine.engineer_clustering_features()
    
    # Step 4: Perform clustering
    kmeans, X_scaled = cluster_engine.perform_clustering(n_clusters=8)
    
    # Step 5: Analyze clusters
    cluster_analysis = cluster_engine.analyze_clusters()
    
    # Step 6: Create intelligent cluster names
    cluster_names, cluster_descriptions = cluster_engine.create_intelligent_cluster_names(cluster_analysis)
    
    # Step 7: Visualize results
    cluster_engine.visualize_clusters(cluster_analysis, cluster_names, cluster_descriptions)
    
    # Step 8: Save everything
    cluster_engine.save_results(cluster_names, cluster_descriptions)
    
    print("\n🎉 PIPELINE COMPLETE!")

if __name__ == "__main__":
    main()
