import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import os
from typing import Tuple, Optional

class CustomerSegmentation:
    """Customer segmentation using RFM analysis and K-Means clustering."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans_model = None
        self.pca_model = None
        self.optimal_k = None
        self.cluster_names = {
            0: 'Casual Buyers',
            1: 'Inactive Customers', 
            2: 'Loyal High-Value Customers',
            3: 'VIP/Wholesale Customers',
            4: 'Frequent Mid-to-High Value Customers'
        }
    
    def prepare_rfm_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare RFM (Recency, Frequency, Monetary) data from transaction data."""
        df_clean = df.copy()
        
        # Handle guest customers
        df_clean['IsGuest'] = (df_clean['CustomerID'].isna() | df_clean['CustomerID'].astype(str).str.startswith('Guest_'))
        df_clean['CustomerID'] = df_clean['CustomerID'].fillna(
            'Guest_' + df_clean.groupby('InvoiceNo').ngroup().astype(str)
        )
        
        # Calculate RFM metrics
        max_date = df_clean['InvoiceDate'].max() + pd.Timedelta(days=1)
        rfm = df_clean.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (max_date - x.max()).days,
            'InvoiceNo': 'nunique',
            'TotalPrice': 'sum',
            'IsGuest': 'first'
        }).rename(columns={
            'InvoiceDate': 'Recency',
            'InvoiceNo': 'Frequency', 
            'TotalPrice': 'Monetary'
        })
        
        return rfm.dropna()
    
    def find_optimal_clusters(self, rfm_data: pd.DataFrame, k_range: range = range(2, 11)) -> Tuple[int, list]:
        """Find optimal number of clusters using silhouette score."""
        X = rfm_data[['Recency', 'Frequency', 'Monetary']].values
        X_scaled = self.scaler.fit_transform(X)
        
        scores = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=1)
            labels = kmeans.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            scores.append(score)
            print(f"K = {k}, Silhouette = {score:.3f}")
        
        optimal_k = k_range[np.argmax(scores)]
        print(f"\nOptimal K = {optimal_k}, Score = {max(scores):.3f}")
        
        return optimal_k, scores
    
    def fit_clustering(self, rfm_data: pd.DataFrame, n_clusters: Optional[int] = None) -> pd.DataFrame:
        """Fit K-Means clustering model and return data with cluster assignments."""
        X = rfm_data[['Recency', 'Frequency', 'Monetary']].values
        X_scaled = self.scaler.fit_transform(X)
        
        if n_clusters is None:
            print("Testing different numbers of clusters...")
            self.optimal_k, _ = self.find_optimal_clusters(rfm_data)
        else:
            self.optimal_k = n_clusters
            print(f"Using specified number of clusters: {n_clusters}")
        
        # Fit final model  
        self.kmeans_model = KMeans(n_clusters=self.optimal_k, random_state=self.random_state, n_init=1)
        clusters = self.kmeans_model.fit_predict(X_scaled)
        
        # Add cluster assignments
        rfm_clustered = rfm_data.copy()
        rfm_clustered['Cluster'] = clusters
        rfm_clustered['Cluster_Name'] = rfm_clustered['Cluster'].map(
            lambda x: self.cluster_names.get(x, f'Cluster_{x}')
        )
        
        # Add PCA coordinates for visualization
        self.pca_model = PCA(n_components=2, random_state=self.random_state)
        coords_2d = self.pca_model.fit_transform(X_scaled)
        rfm_clustered['PC1'] = coords_2d[:, 0]
        rfm_clustered['PC2'] = coords_2d[:, 1]
        
        return rfm_clustered
    
    def predict_cluster(self, rfm_data: pd.DataFrame) -> np.ndarray:
        """Predict clusters for new RFM data."""
        if self.kmeans_model is None:
            raise ValueError("Model not fitted yet. Call fit_clustering first.")
        
        X = rfm_data[['Recency', 'Frequency', 'Monetary']].values
        X_scaled = self.scaler.transform(X)
        return self.kmeans_model.predict(X_scaled)
    
    def get_cluster_summary(self, rfm_clustered: pd.DataFrame) -> pd.DataFrame:
        """Get summary statistics for each cluster."""
        summary = rfm_clustered.groupby('Cluster_Name').agg({
            'Recency': ['mean', 'median'],
            'Frequency': ['mean', 'median'], 
            'Monetary': ['mean', 'median'],
            'IsGuest': ['count', 'sum']
        }).round(2)
        
        # Flatten column names
        summary.columns = ['_'.join(col).strip() for col in summary.columns]
        summary['Guest_Percentage'] = (summary['IsGuest_sum'] / summary['IsGuest_count'] * 100).round(2)
        
        return summary
    
    def visualize_clusters(self, rfm_clustered: pd.DataFrame, save_path: Optional[str] = None):
        """Create cluster visualization plots."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 2D PCA plot
        sns.scatterplot(data=rfm_clustered, x='PC1', y='PC2', hue='Cluster_Name', style='IsGuest', ax=ax1, alpha=0.7)
        ax1.set_title(f'Customer Segmentation (K={self.optimal_k}) - PCA Projection')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Cluster distribution
        cluster_counts = rfm_clustered['Cluster_Name'].value_counts()
        ax2.pie(cluster_counts.values, labels=cluster_counts.index, autopct='%1.1f%%')
        ax2.set_title('Cluster Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main function to run customer segmentation."""
    # Load preprocessed data
    try:
        df = pd.read_csv('data/processed/cleaned_retail_data.csv')
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    except FileNotFoundError:
        print("Error: Please run data_preprocessing.py first to generate cleaned data.")
        return
    
    # Initialize segmentation
    segmentation = CustomerSegmentation()
    
    # Prepare RFM data
    print("Preparing RFM data...")
    rfm_data = segmentation.prepare_rfm_data(df)
    print(f"RFM data prepared: {len(rfm_data)} customers")
    
    # Fit clustering - automatically find optimal K
    print("\nFitting clustering model...")
    rfm_clustered = segmentation.fit_clustering(rfm_data)
    
    # Get cluster summary
    print("\nCluster Summary:")
    summary = segmentation.get_cluster_summary(rfm_clustered)
    print(summary)
    
    # Save results
    os.makedirs('data/processed', exist_ok=True)
    rfm_clustered.to_csv('data/processed/customer_segments.csv', index=True)
    summary.to_csv('data/processed/cluster_summary.csv')
    
    print(f"\nSegmentation complete! Results saved to data/processed/")
    print(f"- Customer segments: customer_segments.csv")
    print(f"- Cluster summary: cluster_summary.csv")


if __name__ == "__main__":
    main()