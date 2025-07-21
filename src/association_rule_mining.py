import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
import os
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class AssociationRuleMining:
    """Association Rule Mining for customer segments using FP-Growth algorithm."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.all_rules = []
        self.optimal_params = {}
        self.cluster_statistics = {}
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
        
        # Set visualization style
        plt.style.use('default')
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
        
        # Color scheme
        self.colors = {
            'regular': '#4ECDC4',
            'accent': '#FFE66D',
            'text': '#2C3E50',
            'background': '#F7F9FC'
        }
        
        # Plot settings
        self.plot_settings = {
            'figure.figsize': (12, 8),
            'figure.dpi': 100,
            'figure.facecolor': self.colors['background'],
            'axes.facecolor': self.colors['background'],
            'axes.grid': True,
            'grid.alpha': 0.3,
            'text.color': self.colors['text']
        }
        plt.rcParams.update(self.plot_settings)
    
    def load_data(self, data_path: str = 'data/processed/cleaned_retail_data.csv',
                  cluster_path: str = 'data/processed/customer_segments.csv') -> pd.DataFrame:
        """Load and merge transaction data with customer segments."""
        try:
            # Load transaction data
            df_clean = pd.read_csv(data_path)
            df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
            
            # Load customer segments
            rfm_data = pd.read_csv(cluster_path, index_col=0)
            cluster_mapping = rfm_data[['Cluster']].reset_index()
            
            # Merge data
            df_with_clusters = df_clean.merge(cluster_mapping, on='CustomerID', how='left')
            df_with_clusters = df_with_clusters.dropna(subset=['Cluster'])
            
            return df_with_clusters
            
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            print("Please ensure data preprocessing and customer segmentation are completed first.")
            return pd.DataFrame()
    
    def get_min_occurrence(self, n_transactions: int) -> int:
        """Determine min_occurrence based on transaction count."""
        if n_transactions < 100:
            return 2
        elif n_transactions < 1000:
            return 3
        else:
            return 5
    
    def sample_large_cluster(self, cluster_data: pd.DataFrame, sample_size: int = 1000, cluster_id: Optional[str] = None) -> pd.DataFrame:
        """Sample cluster if too large."""
        n_transactions = cluster_data['InvoiceNo'].nunique()

        if n_transactions <= 3000:
            return cluster_data
        
        unique_invoices = sorted(cluster_data['InvoiceNo'].unique())
        sampled_invoices = pd.Series(unique_invoices).head(sample_size).values
        sampled_data = cluster_data[cluster_data['InvoiceNo'].isin(sampled_invoices)]
        return sampled_data
    
    def prepare_transactions(self, data: pd.DataFrame, column: str, min_occurrence: int, max_items_per_transaction: int = 50, cluster_id: Optional[str] = None) -> List[List]:
        """Prepare and filter transactions for mining."""
        if column not in data.columns:
            print(f"Error: Column {column} not found!")
            return []
        
        # Build transaction dictionary
        transactions_dict = {}
        for _, row in data[['InvoiceNo', column]].iterrows():
            invoice, item = row['InvoiceNo'], row[column]
            if invoice not in transactions_dict:
                transactions_dict[invoice] = []
            if item not in transactions_dict[invoice]:
                transactions_dict[invoice].append(item)
        
        # Filter by frequency
        item_counts = {}
        for items in transactions_dict.values():
            for item in items:
                item_counts[item] = item_counts.get(item, 0) + 1
        
        # Frequent items based on min_occurrence
        frequent_items = {item for item, count in item_counts.items() if count >= min_occurrence}
        
        # Filter and limit transactions
        filtered_transactions = []
        for items in transactions_dict.values():
            filtered = [item for item in items if item in frequent_items][:max_items_per_transaction]
            if filtered:
                filtered_transactions.append(filtered)
        
        return sorted([sorted(t) for t in filtered_transactions])
    
    def calculate_additional_metrics(self, rules: pd.DataFrame) -> pd.DataFrame:
        """Calculate leverage and conviction metrics."""
        if rules.empty:
            return rules
            
        rules['leverage'] = rules['support'] - (rules['antecedent support'] * rules['consequent support'])
        rules['conviction'] = rules.apply(
            lambda row: ((1 - row['consequent support']) / (1 - row['confidence']))
            if row['confidence'] < 1 else float('inf'), axis=1
        )
        
        # Handle infinite values
        if not rules.empty and (rules['conviction'] != np.inf).any():
            max_non_inf = rules[rules['conviction'] != np.inf]['conviction'].max()
            rules['conviction'] = rules['conviction'].replace([np.inf, -np.inf], max_non_inf)
        else:
            rules['conviction'] = rules['conviction'].replace([np.inf, -np.inf], np.nan).fillna(1.0)
        
        return rules
    
    def generate_association_rules(self, cluster_data: pd.DataFrame, column: str, 
                                 min_support: float, min_confidence: float, min_lift: float,
                                 min_occurrence: int, max_items_per_transaction: int = 50,
                                 cluster_id: Optional[str] = None) -> pd.DataFrame:
        """Generate association rules with given parameters."""
        
        # Prepare transactions
        filtered_transactions = self.prepare_transactions(
            cluster_data, column, min_occurrence, max_items_per_transaction, cluster_id
        )
        
        if len(filtered_transactions) < 10:
            return pd.DataFrame()
        
        # Transaction encoding
        te = TransactionEncoder()
        te_ary = te.fit(filtered_transactions).transform(filtered_transactions)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
        df_encoded = df_encoded.reindex(sorted(df_encoded.columns), axis=1)
        
        # FP-Growth
        frequent_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)
        
        if frequent_itemsets.empty:
            return pd.DataFrame()
        
        # Generate rules
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        
        # Add additional metrics
        rules = self.calculate_additional_metrics(rules)
        
        # Limit rules if too many
        max_rules = 5000
        if len(rules) > max_rules:
            rules = rules.sort_values(['lift', 'confidence', 'support'],
                                    ascending=[False, False, False]).head(max_rules)
        
        # Filter by lift
        rules = rules[rules['lift'] >= min_lift]
        
        # Final sort for reproducible output
        if not rules.empty:
            rules = rules.sort_values(['lift', 'confidence', 'support'], ascending=[False, False, False]).reset_index(drop=True)
        
        return rules
    
    def tune_association_parameters(self, cluster_data: pd.DataFrame, cluster_id: str) -> Tuple[float, float, float]:
        """Perform grid search to find optimal parameters."""
        support_values = [0.03, 0.05]
        confidence_values = [0.3, 0.4]
        lift_values = [1.2, 1.5]
        
        n_transactions = cluster_data['InvoiceNo'].nunique()
        min_occurrence = self.get_min_occurrence(n_transactions)
        
        tuning_results = []
        
        # Grid search
        for min_support in support_values:
            for min_confidence in confidence_values:
                for min_lift in lift_values:
                    try:
                        rules = self.generate_association_rules(
                            cluster_data, 'CategoryExtraction', min_support,
                            min_confidence, min_lift, min_occurrence, cluster_id=cluster_id
                        )
                        
                        if not rules.empty:
                            result = {
                                'min_support': min_support,
                                'min_confidence': min_confidence,
                                'min_lift': min_lift,
                                'num_rules': len(rules),
                                'avg_lift': rules['lift'].mean(),
                                'avg_confidence': rules['confidence'].mean(),
                                'avg_leverage': rules['leverage'].mean()
                            }
                            tuning_results.append(result)
                    except Exception as e:
                        print(f"  Error on {min_support}, {min_confidence}: {e}")
        
        if not tuning_results:
            return 0.03, 0.3, 1.2
        
        # Select optimal parameters
        results_df = pd.DataFrame(tuning_results)
        filtered_results = results_df[
            (results_df['num_rules'] >= 20) & (results_df['num_rules'] <= 1000)
        ]
        
        if not filtered_results.empty:
            optimal = filtered_results.sort_values('avg_lift', ascending=False).iloc[0]
            return optimal['min_support'], optimal['min_confidence'], optimal['min_lift']
        else:
            return 0.03, 0.3, 1.2
    
    def analyze_cluster(self, df_with_clusters: pd.DataFrame, cluster_id: str) -> pd.DataFrame:
        """Run analysis for one cluster."""
        print(f"\nAnalyzing Cluster {cluster_id}")
        
        # Get cluster data
        cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster_id].copy()
        n_transactions = cluster_data['InvoiceNo'].nunique()
        n_customers = cluster_data['CustomerID'].nunique()
        
        # Sample if too large
        cluster_data = self.sample_large_cluster(cluster_data, cluster_id=cluster_id)
        n_transactions = cluster_data['InvoiceNo'].nunique()
        
        # Store cluster statistics
        self.cluster_statistics[cluster_id] = {
            'n_customers': n_customers,
            'n_transactions': n_transactions,
            'n_items': cluster_data['StockCode'].nunique(),
            'avg_items_per_txn': cluster_data.groupby('InvoiceNo').size().mean()
        }
        
        # Parameter tuning
        min_support, min_confidence, min_lift = self.tune_association_parameters(cluster_data, cluster_id)
        self.optimal_params[cluster_id] = (min_support, min_confidence, min_lift)
        print(f"Optimal parameters: support={min_support}, confidence={min_confidence}, lift={min_lift}")
        
        # Generate rules
        min_occurrence = self.get_min_occurrence(n_transactions)
        rules = self.generate_association_rules(
            cluster_data, 'CategoryExtraction', min_support,
            min_confidence, min_lift, min_occurrence, cluster_id=cluster_id
        )
        
        if not rules.empty:
            rules['Cluster'] = cluster_id
            print(f"Found {len(rules)} association rules")
        else:
            print(f"No rules found")
        
        return rules
    
    def run_all_clusters(self, df_with_clusters: pd.DataFrame) -> pd.DataFrame:
        """Run association rule mining for all clusters."""
        print("Starting association rule mining for all clusters...")
        
        unique_clusters = sorted(df_with_clusters['Cluster'].unique())
        print(f"Clusters to analyze: {unique_clusters}")
        
        all_rules = []
        
        for cluster_id in unique_clusters:
            try:
                rules = self.analyze_cluster(df_with_clusters, cluster_id)
                if not rules.empty:
                    all_rules.append(rules.copy())
            except Exception as e:
                print(f"Error analyzing cluster {cluster_id}: {e}")
        
        if all_rules:
            combined_rules = pd.concat(all_rules, ignore_index=True)
            self.all_rules = combined_rules
            print(f"\nTotal rules found: {len(combined_rules)}")
            return combined_rules
        else:
            print("\nNo rules found across all clusters")
            return pd.DataFrame()
    
    def create_network_visualization(self, rules: pd.DataFrame, cluster_id: str, top_n: int = 10):
        """Create network graph from top association rules."""
        if rules.empty:
            return
            
        try:
            top_rules = rules.sort_values('lift', ascending=False).head(top_n)
            G = nx.DiGraph()
            
            for _, row in top_rules.iterrows():
                for ant in row['antecedents']:
                    for con in row['consequents']:
                        G.add_edge(ant, con,
                                 weight=row['lift'],
                                 confidence=row['confidence'],
                                 support=row['support'])
            
            if G.number_of_edges() > 0:
                plt.figure(figsize=(12, 10))
                pos = nx.spring_layout(G, seed=42)
                
                # Draw network
                nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue', alpha=0.8)
                edge_widths = [G[u][v]['weight'] * 0.5 for u, v in G.edges()]
                nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.7, 
                                     edge_color='blue', arrows=True, arrowsize=20)
                nx.draw_networkx_labels(G, pos, font_size=8)
                
                plt.title(f"Association Rules Network - Cluster {cluster_id}", fontsize=14)
                plt.axis('off')
                plt.tight_layout()
                plt.show()
            else:
                print("No edges for network visualization")
                
        except Exception as e:
            print(f"Error creating network visualization: {e}")
    
    def visualize_cluster_comparison(self, rules_df: pd.DataFrame, save_path: Optional[str] = None):
        """Create visualization of cluster comparison."""
        if rules_df.empty:
            return
            
        plt.figure(figsize=(16, 12))
        
        # Rule count distribution
        plt.subplot(2, 3, 1)
        cluster_counts = rules_df['Cluster'].value_counts()
        plt.bar(range(len(cluster_counts)), cluster_counts.values, color='cornflowerblue')
        plt.title('Rules Distribution by Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Number of Rules')
        plt.xticks(range(len(cluster_counts)), [f'Cluster {x}' for x in cluster_counts.index])
        
        # Average lift by cluster
        plt.subplot(2, 3, 2)
        avg_lifts = rules_df.groupby('Cluster')['lift'].mean()
        plt.bar(range(len(avg_lifts)), avg_lifts.values, color='lightgreen')
        plt.title('Average Lift by Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Average Lift')
        plt.xticks(range(len(avg_lifts)), [f'Cluster {x}' for x in avg_lifts.index])
        
        # Support vs Confidence scatter
        plt.subplot(2, 3, 3)
        for cluster in rules_df['Cluster'].unique():
            cluster_data = rules_df[rules_df['Cluster'] == cluster]
            plt.scatter(cluster_data['support'], cluster_data['confidence'], 
                       label=f'Cluster {cluster}', alpha=0.6)
        plt.xlabel('Support')
        plt.ylabel('Confidence')
        plt.title('Support vs Confidence by Cluster')
        plt.legend()
        
        # Lift distribution
        plt.subplot(2, 3, 4)
        cluster_data_for_violin = []
        cluster_labels_for_violin = []
        for cluster in rules_df['Cluster'].unique():
            cluster_data_for_violin.extend(rules_df[rules_df['Cluster'] == cluster]['lift'].values)
            cluster_labels_for_violin.extend([f'Cluster {cluster}'] * 
                                           len(rules_df[rules_df['Cluster'] == cluster]))
        
        violin_df = pd.DataFrame({'Cluster': cluster_labels_for_violin, 'Lift': cluster_data_for_violin})
        sns.violinplot(data=violin_df, x='Cluster', y='Lift')
        plt.title('Lift Distribution by Cluster')
        plt.xticks(rotation=45)
        
        # Top rules summary
        plt.subplot(2, 3, 5)
        top_rules_per_cluster = rules_df.groupby('Cluster')['lift'].max()
        plt.bar(range(len(top_rules_per_cluster)), top_rules_per_cluster.values, color='orange')
        plt.title('Highest Lift Rule by Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Max Lift')
        plt.xticks(range(len(top_rules_per_cluster)), [f'Cluster {x}' for x in top_rules_per_cluster.index])
        
        plt.suptitle('Association Rules Analysis Across All Clusters', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_cluster_summary(self, rules_df: pd.DataFrame) -> pd.DataFrame:
        """Get summary statistics for each cluster."""
        if rules_df.empty:
            return pd.DataFrame()
            
        summary = rules_df.groupby('Cluster').agg({
            'support': ['count', 'mean', 'std', 'min', 'max'],
            'confidence': ['mean', 'std', 'min', 'max'],
            'lift': ['mean', 'std', 'min', 'max'],
            'leverage': ['mean', 'std'],
            'conviction': ['mean', 'std']
        }).round(4)
        
        # Flatten column names
        summary.columns = [f'{col[1]}_{col[0]}' if col[1] != '' else col[0] for col in summary.columns]
        summary = summary.rename(columns={'count_support': 'total_rules'})
        
        return summary
    
    def save_results(self, rules_df: pd.DataFrame, output_dir: str = 'data/processed'):
        """Save all results to CSV files."""
        os.makedirs(output_dir, exist_ok=True)
        
        if not rules_df.empty:
            rules_df.to_csv(os.path.join(output_dir, 'association_rules.csv'), index=False)
            print(f"Results saved to {output_dir}/")

def main():
    """Main function to run association rule mining."""
    # Initialize association rule mining
    arm = AssociationRuleMining()
    
    # Load data
    df_with_clusters = arm.load_data()
    
    if df_with_clusters.empty:
        print("No data loaded. Please check data paths.")
        return
    
    # Run association rule mining for all clusters
    all_rules = arm.run_all_clusters(df_with_clusters)
    
    if not all_rules.empty:
        # Get summary
        print("\nCluster Summary:")
        summary = arm.get_cluster_summary(all_rules)
        if not summary.empty:
            print(summary)
        
        # Save results
        arm.save_results(all_rules)
        
        # Create visualizations
        arm.visualize_cluster_comparison(all_rules, 'data/processed/association_rules_comparison.png')
        
        # Create network visualizations for each cluster
        for cluster_id in all_rules['Cluster'].unique():
            cluster_rules = all_rules[all_rules['Cluster'] == cluster_id]
            if not cluster_rules.empty:
                arm.create_network_visualization(cluster_rules, cluster_id)
        
        print(f"\nAssociation rule mining complete!")
        print(f"Total rules found: {len(all_rules)}")
        print(f"\nResults saved to data/processed/")
    else:
        print("\nNo association rules found.")

if __name__ == "__main__":
    main()