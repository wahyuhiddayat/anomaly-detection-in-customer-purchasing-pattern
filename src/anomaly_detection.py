import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import shap
from scipy.stats import spearmanr
import logging
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class AnomalyDetection:
    """Anomaly detection for association rules using Isolation Forest and LOF."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.isolation_forest_models = {}
        self.lof_models = {}
        self.scalers = {}
        self.feature_importance_results = {}
        self.cluster_strategies = {
            0: {
                'name': 'Casual Buyers',
                'strategies': [
                    ('cross_selling', 'Offer complementary souvenirs'),
                    ('bundling', 'Create affordable bundles'),
                    ('promotions', 'Use time-limited offers for popular items')
                ],
                'fallback_strategies': [('personalization', 'Offer tailored recommendations')]
            },
            1: {
                'name': 'Inactive Customers',
                'strategies': [
                    ('reactivation', 'Re-engage with targeted offers'),
                    ('cross_selling', 'Offer complementary products'),
                    ('discounts', 'Provide special discounts')
                ],
                'fallback_strategies': [('promotions', 'Use time-limited offers')]
            },
            2: {
                'name': 'Loyal High-Value Customers',
                'strategies': [
                    ('upselling', 'Offer premium products'),
                    ('personalization', 'Offer tailored recommendations'),
                    ('loyalty_program', 'Enroll in loyalty rewards')
                ],
                'fallback_strategies': [
                    ('cross_selling', 'Offer complementary products'),
                    ('promotions', 'Exclusive offers for loyal customers')
                ]
            },
            3: {
                'name': 'VIP/Wholesale Customers',
                'strategies': [
                    ('wholesale', 'Offer bulk purchase deals'),
                    ('cross_selling', 'Offer complementary products'),
                    ('special_services', 'Provide dedicated account management')
                ],
                'fallback_strategies': [('promotions', 'Exclusive bulk offers')]
            },
            4: {
                'name': 'Frequent Mid-to-High Value Customers',
                'strategies': [
                    ('cross_selling', 'Offer complementary products'),
                    ('upselling', 'Offer premium products'),
                    ('seasonal_promotions', 'Offer seasonal deals')
                ],
                'fallback_strategies': [('personalization', 'Offer tailored recommendations')]
            }
        }
        
        # Setup logging
        logging.basicConfig(level=logging.WARNING)
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
    
    def preprocess_rules(self, rules_df: pd.DataFrame, features_to_normalize: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
        """
        Preprocess association rules DataFrame by resetting index and normalizing features.

        Parameters:
        -----------
        rules_df : pandas DataFrame
            DataFrame containing association rules with metrics
        features_to_normalize : list, optional
            List of column names to normalize. If None, defaults to standard association rule metrics.

        Returns:
        --------
        rules_df : pandas DataFrame
            DataFrame with reset index and additional normalized feature columns
        features_scaled : pandas DataFrame
            DataFrame containing normalized features
        scaler : StandardScaler
            Trained StandardScaler object for inverse transformation if needed
        """
        # Default features to normalize if not specified
        if features_to_normalize is None:
            features_to_normalize = [
                'antecedent support', 'consequent support', 'support',
                'confidence', 'lift', 'leverage', 'conviction'
            ]

        # Reset index to ensure consistent row ordering
        rules_df = rules_df.reset_index(drop=True)

        # Validate that required columns exist
        missing_cols = [col for col in features_to_normalize if col not in rules_df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in rules_df: {missing_cols}")

        # Extract features for normalization
        features = rules_df[features_to_normalize].copy()

        # Handle undefined or extreme values (e.g., inf, -inf, NaN)
        features = features.replace([np.inf, -np.inf], np.nan)
        # Fill NaN with median for each column to preserve distribution
        for col in features.columns:
            median_value = features[col].median()
            features[col] = features[col].fillna(median_value)

        # Initialize and apply StandardScaler
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        features_scaled = pd.DataFrame(
            features_scaled,
            columns=[f"{col}_normalized" for col in features_to_normalize],
            index=rules_df.index
        )

        # Merge normalized features back into rules_df
        rules_df = pd.concat([rules_df, features_scaled], axis=1)

        # Add rule_id column and sort the rules by rule_id
        rules_df = self._add_rule_id_column(rules_df)
        rules_df = rules_df.sort_values(by='rule_id').reset_index(drop=True)

        return rules_df, features_scaled, scaler
    
    def _add_rule_id_column(self, rules_df: pd.DataFrame) -> pd.DataFrame:
        """Add rule_id column for consistent identification."""
        rules_df = rules_df.copy()
        rules_df["rule_id"] = rules_df.apply(
            lambda row: f"{sorted(row['antecedents'])}->{sorted(row['consequents'])}",
            axis=1
        )
        return rules_df

    def detect_anomalous_rules(self, rules_df: pd.DataFrame, contamination: float = 0.05, n_neighbors: int = 20, cluster_id: Optional[int] = None) -> Tuple[pd.DataFrame, Optional[IsolationForest], Optional[pd.DataFrame]]:
        """
        Apply Isolation Forest and LOF to detect anomalous rules.

        Parameters:
        -----------
        rules_df : pandas DataFrame
            DataFrame containing association rules with normalized metrics.
        contamination : float, default=0.05
            Expected proportion of anomalies for both algorithms.
        n_neighbors : int, default=20
            Number of neighbors for LOF algorithm.
        cluster_id : int, optional
            Cluster identifier for model storage

        Returns:
        --------
        rules_df : pandas DataFrame
            Original DataFrame with added columns:
            - 'isolation_forest_anomaly': 1 for anomalies, 0 for inliers.
            - 'lof_anomaly': 1 for anomalies, 0 for inliers.
            - 'combined_anomaly': 1 if both algorithms detect anomaly, 0 otherwise.
            - 'anomaly_score': Isolation Forest anomaly score (negative = anomaly).
        iso_forest : IsolationForest
            Trained Isolation Forest model.
        features : pandas DataFrame
            Feature matrix used for anomaly detection.
        """
        if len(rules_df) < 10:
            print(f"Not enough rules ({len(rules_df)}) for anomaly detection")
            rules_df['isolation_forest_anomaly'] = 0
            rules_df['lof_anomaly'] = 0
            rules_df['combined_anomaly'] = 0
            rules_df['anomaly_score'] = 0.0
            return rules_df, None, None

        # Select normalized features
        feature_columns = [
            'antecedent support_normalized', 'consequent support_normalized', 'support_normalized',
            'confidence_normalized', 'lift_normalized', 'leverage_normalized', 'conviction_normalized'
        ]
        features = rules_df[feature_columns].copy()

        # Fill NaN with median to avoid skewing
        features = features.fillna(features.median())

        # Isolation Forest
        iso_forest = IsolationForest(n_estimators=100, contamination=contamination, random_state=self.random_state, n_jobs=1)
        iso_forest_preds = iso_forest.fit_predict(features)
        rules_df['isolation_forest_anomaly'] = np.where(iso_forest_preds == -1, 1, 0)

        # Local Outlier Factor
        lof = LocalOutlierFactor(n_neighbors=min(n_neighbors, len(features) - 1), contamination='auto')
        lof_preds = lof.fit_predict(features)
        rules_df['lof_anomaly'] = np.where(lof_preds == -1, 1, 0)

        # Combined anomaly (both algorithms agree)
        rules_df['combined_anomaly'] = (rules_df['isolation_forest_anomaly'] + rules_df['lof_anomaly'] > 1).astype(int)

        # Anomaly score (negative = anomaly, per scikit-learn convention)
        rules_df['anomaly_score'] = iso_forest.decision_function(features)

        # Store models if cluster_id is provided
        if cluster_id is not None:
            self.isolation_forest_models[cluster_id] = iso_forest
            self.lof_models[cluster_id] = lof

        return rules_df, iso_forest, features

    def analyze_iforest_contamination(self, rules_df: pd.DataFrame, cluster_id: int, contamination_rates: List[float] = [0.01, 0.03, 0.05, 0.07, 0.1]) -> Dict[str, Any]:
        """
        Analyze Isolation Forest contamination rates for a specific cluster of rules.

        Parameters:
        -----------
        rules_df : pandas DataFrame
            DataFrame containing association rules with normalized metrics
        cluster_id : int or string
            The identifier of a cluster (e.g., 0 for rules_cluster0, "Guest" for rules_cluster_guest)
        contamination_rates : list, optional
            List of contamination rates to test

        Returns:
        --------
        dict
            Dictionary containing the analysis results
        """
        # Set global random seed for NumPy
        np.random.seed(self.random_state)

        numerical_features = [
            'antecedent support_normalized', 'consequent support_normalized', 'support_normalized',
            'confidence_normalized', 'lift_normalized', 'leverage_normalized', 'conviction_normalized'
        ]
        X = rules_df[numerical_features]

        # Store predictions
        all_predictions = {}

        # Fit the model once since scores won't change
        iforest = IsolationForest(random_state=self.random_state)
        iforest.fit(X)
        base_scores = iforest.score_samples(X)

        for rate in contamination_rates:
            threshold = np.percentile(base_scores, rate * 100)
            # Anomaly = low score, show that the point is easy to be separated from the rest
            predictions = (base_scores <= threshold).astype(int)
            all_predictions[rate] = predictions

        # Calculate Jaccard similarities
        jaccard_similarities = []
        for i in range(len(contamination_rates)-1):
            rate1, rate2 = contamination_rates[i], contamination_rates[i+1]
            pred1 = set(np.where(all_predictions[rate1] == 1)[0])
            pred2 = set(np.where(all_predictions[rate2] == 1)[0])

            intersection = len(pred1.intersection(pred2))
            union = len(pred1.union(pred2))
            jaccard = intersection / union if union > 0 else 1.0
            jaccard_similarities.append(jaccard)

        # Create visualization
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Score distribution with different thresholds
        ax1.hist(base_scores, bins=30, density=True)

        colors = ['red', 'blue', 'green', 'purple', 'orange']
        styles = ['--', '-.', ':', '-', '--']

        thresholds = {}
        for rate, color, style in zip(contamination_rates, colors, styles):
            threshold = np.percentile(base_scores, rate * 100)
            thresholds[rate] = threshold
            ax1.axvline(x=threshold, color=color, linestyle=style,
                        label=f'threshold at {rate}')

        ax1.set_xlabel('Anomaly Score')
        ax1.set_ylabel('Density')
        ax1.set_title(f'Distribution of Anomaly Scores with Different Thresholds (Cluster {cluster_id})')
        ax1.legend()
        ax1.grid(True)

        # Plot 2: Jaccard similarity
        ax2.plot(contamination_rates[1:], jaccard_similarities, marker='o')
        ax2.set_xlabel('Contamination Rate')
        ax2.set_ylabel('Jaccard Similarity with Previous Rate')
        ax2.set_title(f'Stability of Binary Predictions with Different Contamination Rates (Cluster {cluster_id})')
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

        # Print statistics
        print(f"\nAnalysis for Cluster {cluster_id}")
        print("\nThreshold values for each contamination rate:")
        for rate in contamination_rates:
            threshold = thresholds[rate]
            print(f"Rate {rate}: threshold = {threshold:.4f}")

        anomaly_counts = {}
        print("\nNumber of anomalies detected at each contamination rate:")
        for rate in contamination_rates:
            n_anomalies = sum(all_predictions[rate] == 1)
            anomaly_counts[rate] = n_anomalies
            print(f"Rate {rate}: {n_anomalies} anomalies")

        return {
            'base_scores': base_scores,
            'predictions': all_predictions,
            'jaccard_similarities': jaccard_similarities,
            'thresholds': thresholds,
            'anomaly_counts': anomaly_counts
        }

    def analyze_lof_parameters(self, rules_df: pd.DataFrame, cluster_id: int, n_neighbors_values: List[int] = [5, 10, 20, 30, 50]) -> Dict[str, Any]:
        """
        Analyze LOF parameters for a specific cluster of rules.

        Parameters:
        -----------
        rules_df : pandas DataFrame
            DataFrame containing association rules with normalized metrics
        cluster_id : int or string
            The identifier of a cluster (e.g., 0 for rules_cluster0, "Guest" for rules_cluster_guest)
        n_neighbors_values : list, optional
            List of k values to test for LOF

        Returns:
        --------
        dict
            Dictionary containing the analysis results
        """
        features = ['antecedent support_normalized', 'consequent support_normalized', 'support_normalized',
                   'confidence_normalized', 'lift_normalized', 'leverage_normalized', 'conviction_normalized']

        # Store scores for each k
        all_scores = {}

        for k in n_neighbors_values:
            lof = LocalOutlierFactor(n_neighbors=k, contamination='auto')
            # First fit the model
            lof.fit(rules_df[features])
            # Get the actual LOF scores
            scores = lof.negative_outlier_factor_
            all_scores[k] = scores

        # Create a figure with multiple subplots
        _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

        # Plot 1: Score distributions for different k values
        for k in n_neighbors_values:
            ax1.hist(all_scores[k], bins=30, alpha=0.5, label=f'k={k}', density=True)
        ax1.set_xlabel('LOF Score')
        ax1.set_ylabel('Density')
        ax1.set_title(f'Distribution of LOF Scores for Different k Values (Cluster {cluster_id})')
        ax1.legend()
        ax1.grid(True)

        # Plot 2: Score correlation between consecutive k values
        correlations = []
        for i in range(len(n_neighbors_values)-1):
            k1, k2 = n_neighbors_values[i], n_neighbors_values[i+1]
            corr = spearmanr(all_scores[k1], all_scores[k2])[0]
            correlations.append(corr)

        ax2.plot(n_neighbors_values[:-1], correlations, marker='o')
        ax2.set_xlabel('k')
        ax2.set_ylabel('Spearman Correlation with next k value')
        ax2.set_title(f'Score Stability Across k Values (Cluster {cluster_id})')
        ax2.grid(True)

        # Plot 3: Boxplot of scores for each k
        box_data = [all_scores[k] for k in n_neighbors_values]
        ax3.boxplot(box_data, labels=n_neighbors_values)
        ax3.set_xlabel('k value')
        ax3.set_ylabel('LOF Score')
        ax3.set_title(f'LOF Score Distribution by k (Cluster {cluster_id})')
        ax3.grid(True)

        plt.tight_layout()
        plt.show()

        # Print detailed statistics
        print(f"\nScore Statistics for Cluster {cluster_id}:")
        stats = {}
        for k in n_neighbors_values:
            scores = all_scores[k]
            print(f"\nk = {k}:")
            print(f"Mean score: {np.mean(scores):.4f}")
            print(f"Std score: {np.std(scores):.4f}")
            print(f"Min score (strongest outlier): {np.min(scores):.4f}")
            print(f"Max score (strongest inlier): {np.max(scores):.4f}")

            # Calculate percentage of potential outliers (scores < -1)
            outliers_percent = (scores < -1).mean() * 100
            print(f"Percentage of points with score < -1: {outliers_percent:.2f}%")

            stats[k] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'outliers_percent': outliers_percent
            }

        # Calculate rank correlation
        print(f"\nSpearman Rank Correlations between consecutive k values (Cluster {cluster_id}):")
        rank_correlations = {}
        for i in range(len(n_neighbors_values)-1):
            k1, k2 = n_neighbors_values[i], n_neighbors_values[i+1]
            rank_corr = spearmanr(all_scores[k1], all_scores[k2])[0]
            print(f"k={k1} vs k={k2}: {rank_corr:.4f}")
            rank_correlations[f'{k1}_{k2}'] = rank_corr

        return {
            'scores': all_scores,
            'correlations': correlations,
            'statistics': stats,
            'rank_correlations': rank_correlations
        }

    def explain_anomalies(self, rules_df: pd.DataFrame, model: IsolationForest, features: pd.DataFrame, cluster_id: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Compute feature importance ranking for anomaly detection using SHAP

        Parameters:
        -----------
        rules_df : pandas DataFrame
            DataFrame with anomaly detection results
        model : fitted Isolation Forest model
            The trained Isolation Forest model
        features : pandas DataFrame
            Feature matrix used for anomaly detection
        cluster_id : int, optional
            Cluster identifier for storing results

        Returns:
        --------
        feature_importance : pandas DataFrame
            DataFrame with features ranked by their importance to anomaly detection
        """
        anomalies = rules_df[rules_df['isolation_forest_anomaly'] == 1]

        if len(anomalies) == 0:
            print("No anomalies detected")
            return None

        anomaly_indices = anomalies.index
        anomaly_features = features.loc[anomaly_indices]

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(anomaly_features)

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = pd.DataFrame({
            'feature': features.columns,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)

        # Store feature importance if cluster_id is provided
        if cluster_id is not None:
            self.feature_importance_results[cluster_id] = feature_importance

        return feature_importance

    def display_anomaly_results(self, rules_with_anomalies: pd.DataFrame, feature_importance: Optional[pd.DataFrame], cluster_id: Optional[int] = None):
        """
        Display anomaly detection results and feature importance ranking

        Parameters:
        -----------
        rules_with_anomalies : pandas DataFrame
            DataFrame with anomaly detection results
        feature_importance : pandas DataFrame or None
            Feature importance ranking from explain_anomalies
        cluster_id : int or None, optional
            Cluster identifier for labeling output
        """
        label = f"Cluster {cluster_id}" if cluster_id is not None else "Rules"
        print(f"\nProcessing {label} with {len(rules_with_anomalies)} rules...")

        iso_anomalies = rules_with_anomalies['isolation_forest_anomaly'].sum()
        lof_anomalies = rules_with_anomalies['lof_anomaly'].sum()
        combined_anomalies = rules_with_anomalies['combined_anomaly'].sum()

        print(f"Isolation Forest detected: {iso_anomalies} anomalies")
        print(f"LOF detected: {lof_anomalies} anomalies")
        print(f"Both methods agree on: {combined_anomalies} anomalies")

        if feature_importance is not None:
            print(f"\nFeature Importance Ranking for Anomalies in {label}:")
            print(feature_importance.to_string(index=False))

            plt.figure(figsize=(10, 6))
            plt.bar(feature_importance['feature'], feature_importance['importance'])
            plt.xticks(rotation=45, ha='right')
            plt.xlabel('Features')
            plt.ylabel('Mean Absolute SHAP Value')
            plt.title(f'Feature Importance for Anomalous Rules in {label}')
            plt.tight_layout()
            plt.show()

    def visualize_anomaly_methods(self, rules_df: pd.DataFrame, x_metric: str = 'support', y_metric: str = 'lift', cluster_id: Optional[int] = None):
        """
        Visualize rules detected as anomalies by Isolation Forest, LOF, or both.

        Parameters:
        -----------
        rules_df : pandas DataFrame
            DataFrame with anomaly detection results from detect_anomalous_rules
        x_metric : str
            Metric for x-axis (default: 'support')
        y_metric : str
            Metric for y-axis (default: 'lift')
        cluster_id : int or None, optional
            Cluster identifier for labeling the plot
        """
        # Ensure required columns exist
        required_cols = ['isolation_forest_anomaly', 'lof_anomaly', x_metric, y_metric]
        if not all(col in rules_df.columns for col in required_cols):
            print("Required columns missing from rules_df")
            return

        # Create figure
        plt.figure(figsize=(10, 6))

        # Define categories based on anomaly flags
        normal = rules_df[(rules_df['isolation_forest_anomaly'] == 0) &
                          (rules_df['lof_anomaly'] == 0)]
        iso_only = rules_df[(rules_df['isolation_forest_anomaly'] == 1) &
                            (rules_df['lof_anomaly'] == 0)]
        lof_only = rules_df[(rules_df['isolation_forest_anomaly'] == 0) &
                            (rules_df['lof_anomaly'] == 1)]
        both = rules_df[(rules_df['isolation_forest_anomaly'] == 1) &
                        (rules_df['lof_anomaly'] == 1)]

        # Plot each category with different markers/colors
        plt.scatter(normal[x_metric], normal[y_metric],
                    c='gray', alpha=0.5, label='Normal Rules', marker='o')
        plt.scatter(iso_only[x_metric], iso_only[y_metric],
                    c='blue', label='Isolation Forest Only', marker='x', s=100)
        plt.scatter(lof_only[x_metric], lof_only[y_metric],
                    c='green', label='LOF Only', marker='+', s=100)
        plt.scatter(both[x_metric], both[y_metric],
                    c='red', label='Both Methods', marker='*', s=150)

        label = f"Cluster {cluster_id}" if cluster_id is not None else "Rules"
        plt.xlabel(x_metric.capitalize())
        plt.ylabel(y_metric.capitalize())
        plt.title(f'Anomaly Detection Comparison for {label}')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        plt.show()

    def extract_items_from_rules(self, rules_df: pd.DataFrame, anomaly_columns: List[str] = ['isolation_forest_anomaly', 'lof_anomaly', 'combined_anomaly']) -> Dict[str, pd.DataFrame]:
        """Extract items contributing to anomalous rules."""
        result_dfs = {}

        for anomaly_col in anomaly_columns:
            anomalous_rules = rules_df[rules_df[anomaly_col] == 1]
            normal_rules = rules_df[rules_df[anomaly_col] == 0]

            anomalous_items = {}
            normal_items = {}

            for _, rule in anomalous_rules.iterrows():
                items = list(rule['antecedents']) + list(rule['consequents'])
                for item in items:
                    anomalous_items[item] = anomalous_items.get(item, 0) + 1

            for _, rule in normal_rules.iterrows():
                items = list(rule['antecedents']) + list(rule['consequents'])
                for item in items:
                    normal_items[item] = normal_items.get(item, 0) + 1

            item_ratios = {}
            n_anomalous = len(anomalous_rules)
            n_normal = len(normal_rules)

            all_items = set(anomalous_items.keys()).union(normal_items.keys())
            for item in all_items:
                freq_anomalous = anomalous_items.get(item, 0) / n_anomalous if n_anomalous > 0 else 0
                freq_normal = normal_items.get(item, 0) / n_normal if n_normal > 0 else 0
                ratio = freq_anomalous / freq_normal if freq_normal > 0 else float('inf')
                item_ratios[item] = {'anomalous_freq': freq_anomalous, 'normal_freq': freq_normal, 'ratio': ratio}

            result_dfs[anomaly_col] = pd.DataFrame.from_dict(item_ratios, orient='index').sort_values('ratio', ascending=False)

        return result_dfs

    def find_business_rules(self, rules_df: pd.DataFrame, feature_importance: Optional[pd.DataFrame] = None, cluster_id: Optional[int] = None, top_n: int = 5,
                            min_support: Optional[float] = None, min_confidence: Optional[float] = None, use_initial_filters: bool = True) -> Optional[pd.DataFrame]:
        """Find and recommend business rules based on anomaly detection results."""
        df = rules_df.copy()
        business_rules = pd.DataFrame()

        # Filter by cluster_id if provided
        if cluster_id is not None:
            df = df[df['Cluster'] == cluster_id]
            if df.empty:
                logging.warning(f"No rules found for Cluster {cluster_id}")
                return None

        # Apply initial filters if available
        if use_initial_filters and 'min_support' in rules_df.attrs and 'min_confidence' in rules_df.attrs:
            df = df[(df['support'] >= rules_df.attrs['min_support']) &
                    (df['confidence'] >= rules_df.attrs['min_confidence'])]
        elif min_support is not None and min_confidence is not None:
            df = df[(df['support'] >= min_support) & (df['confidence'] >= min_confidence)]

        # Filter for anomalous rules only if feature_importance is provided
        if feature_importance is not None:
            df = df[(df['isolation_forest_anomaly'] == 1) | (df['lof_anomaly'] == 1)]
            if df.empty:
                logging.warning(f"No anomalous rules found for Cluster {cluster_id} with given thresholds")
                return None

        # Determine key metrics
        if feature_importance is not None:
            key_metrics = feature_importance.sort_values('importance', ascending=False)['feature'].head(2).tolist()
            key_metrics = [m + '_normalized' if m + '_normalized' in df.columns else m for m in key_metrics]
        else:
            # Define default metrics
            key_metrics = [m for m in ['support_normalized', 'confidence_normalized', 'lift_normalized',
                                       'support', 'confidence', 'lift'] if m in df.columns][:3]

        if not key_metrics:
            logging.warning("No valid metrics found in DataFrame")
            return None

        # Calculate composite score
        df['composite_score'] = df[key_metrics].mean(axis=1)
        df = df.sort_values('composite_score', ascending=False)

        # Assign key_metric based on highest value for each rule if no feature_importance
        if feature_importance is None:
            def get_key_metric(row):
                metrics = {m: row[m] for m in key_metrics}
                return max(metrics, key=metrics.get)

            df['key_metric'] = df.apply(get_key_metric, axis=1)
        else:
            df['key_metric'] = key_metrics[0]  # Use primary metric from SHAP

        # Process rules and assign strategies
        strategy_recommendations = []
        for cluster in df['Cluster'].unique():
            cluster_df = df[df['Cluster'] == cluster].head(top_n)
            cluster_info = self.cluster_strategies.get(cluster, {'name': 'Unknown', 'strategies': [], 'fallback_strategies': []})

            for _, row in cluster_df.iterrows():
                key_metric = row['key_metric'] if feature_importance is None else key_metrics[0]
                metric_value = row[key_metric]
                recommended_strategy = None
                strategy_desc = None

                # Try primary strategies
                if key_metric in ['support', 'support_normalized']:
                    recommended_strategy = next((s for s, desc in cluster_info['strategies']
                                                if s in ['bundling', 'promotions', 'wholesale']), None)
                    if recommended_strategy:
                        strategy_desc = next((desc for s, desc in cluster_info['strategies']
                                             if s == recommended_strategy), '')
                elif key_metric in ['leverage', 'leverage_normalized']:
                    recommended_strategy = next((s for s, desc in cluster_info['strategies']
                                                if s in ['cross_selling', 'wholesale']), None)
                    if recommended_strategy:
                        strategy_desc = next((desc for s, desc in cluster_info['strategies']
                                             if s == recommended_strategy), '')
                elif key_metric in ['conviction', 'conviction_normalized']:
                    recommended_strategy = next((s for s, desc in cluster_info['strategies']
                                                if s in ['personalization', 'reactivation', 'upselling']), None)
                    if recommended_strategy:
                        strategy_desc = next((desc for s, desc in cluster_info['strategies']
                                             if s == recommended_strategy), '')
                elif key_metric in ['confidence', 'confidence_normalized', 'lift', 'lift_normalized']:
                    recommended_strategy = next((s for s, desc in cluster_info['strategies']
                                                if s in ['cross_selling', 'promotions']), None)
                    if recommended_strategy:
                        strategy_desc = next((desc for s, desc in cluster_info['strategies']
                                             if s == recommended_strategy), '')

                # Fallback to alternative strategies if no match
                if not recommended_strategy:
                    if key_metric in ['support', 'support_normalized']:
                        recommended_strategy = next((s for s, desc in cluster_info['fallback_strategies']
                                                    if s in ['personalization', 'cross_selling', 'promotions']), None)
                    elif key_metric in ['leverage', 'leverage_normalized', 'confidence', 'confidence_normalized',
                                       'lift', 'lift_normalized']:
                        recommended_strategy = next((s for s, desc in cluster_info['fallback_strategies']
                                                    if s in ['cross_selling', 'promotions']), None)
                    if recommended_strategy:
                        strategy_desc = next((desc for s, desc in cluster_info['fallback_strategies']
                                             if s == recommended_strategy), '')
                    else:
                        logging.warning(f"No suitable strategy found for Cluster {cluster} with key_metric {key_metric}")

                if recommended_strategy:
                    strategy_recommendations.append({
                        'Cluster': cluster,
                        'Cluster_Name': cluster_info['name'],
                        'Rule': row['rule_str'],
                        'Antecedents': row['antecedents'],
                        'Consequents': row['consequents'],
                        'Support': row['support'],
                        'Confidence': row['confidence'],
                        'Lift': row['lift'],
                        'Leverage': row['leverage'],
                        'Conviction': row['conviction'],
                        'Key_Metric': key_metric,
                        'Key_Metric_Value': metric_value,
                        'Anomaly_Score': row['anomaly_score'],
                        'Recommended_Strategy': recommended_strategy,
                        'Strategy_Description': strategy_desc
                    })

        business_rules = pd.DataFrame(strategy_recommendations)
        if business_rules.empty:
            logging.warning("No beneficial rules identified with given criteria")
            return None

        return business_rules.sort_values(['Cluster', 'Key_Metric_Value'], ascending=[True, False])

    def predict_anomalies(self, rules_df: pd.DataFrame, cluster_id: int) -> pd.DataFrame:
        """Predict anomalies using trained models."""
        if cluster_id not in self.isolation_forest_models:
            raise ValueError(f"No trained model found for cluster {cluster_id}")
        
        # Preprocess the data
        rules_processed, _, _ = self.preprocess_rules(rules_df)
        
        # Select features
        feature_columns = [
            'antecedent support_normalized', 'consequent support_normalized', 'support_normalized',
            'confidence_normalized', 'lift_normalized', 'leverage_normalized', 'conviction_normalized'
        ]
        features = rules_processed[feature_columns].fillna(rules_processed[feature_columns].median())
        
        # Predict using saved model
        iso_forest = self.isolation_forest_models[cluster_id]
        predictions = iso_forest.predict(features)
        anomaly_scores = iso_forest.decision_function(features)
        
        # Add results to dataframe
        rules_processed['isolation_forest_anomaly'] = np.where(predictions == -1, 1, 0)
        rules_processed['anomaly_score'] = anomaly_scores
        
        return rules_processed

def main():
    """Complete anomaly detection workflow following the exact steps from the original analysis."""
    # Initialize the anomaly detection system
    anomaly_detector = AnomalyDetection(random_state=42)
    
    try:
        # Load real association rules data
        print("Loading association rules data...")
        rules_df = pd.read_csv('data/processed/association_rules.csv')
        
        # Convert string representations of frozensets back to actual frozensets
        rules_df['antecedents'] = rules_df['antecedents'].apply(eval)
        rules_df['consequents'] = rules_df['consequents'].apply(eval)
        
        # Create rule_str column for business rules
        rules_df['rule_str'] = rules_df.apply(
            lambda row: f"{list(row['antecedents'])} -> {list(row['consequents'])}", axis=1
        )
        
        print(f"Loaded {len(rules_df)} association rules")
        print(f"Clusters found: {sorted(rules_df['Cluster'].unique())}")
        
        # Split rules by cluster
        rules_cluster0 = rules_df[rules_df['Cluster'] == 0].copy()
        rules_cluster1 = rules_df[rules_df['Cluster'] == 1].copy() 
        rules_cluster2 = rules_df[rules_df['Cluster'] == 2].copy()
        rules_cluster3 = rules_df[rules_df['Cluster'] == 3].copy()
        rules_cluster4 = rules_df[rules_df['Cluster'] == 4].copy()
        
        print(f"Cluster 0: {len(rules_cluster0)} rules")
        print(f"Cluster 1: {len(rules_cluster1)} rules") 
        print(f"Cluster 2: {len(rules_cluster2)} rules")
        print(f"Cluster 3: {len(rules_cluster3)} rules")
        print(f"Cluster 4: {len(rules_cluster4)} rules")
        
        # ANOMALY DETECTION FOR CLUSTER 0
        print("\n" + "="*50)
        print("ANOMALY DETECTION FOR CLUSTER 0")
        print("="*50)
        
        # Preprocess rules
        rules0_df, features0_scaled, scaler0 = anomaly_detector.preprocess_rules(rules_cluster0.copy())
        print(f"Cluster 0: Processed {len(rules0_df)} rules")
        
        # Analyze LOF parameters
        anomaly_detector.analyze_lof_parameters(rules0_df.copy(), 0)
        
        # Detect anomalous rules
        rules_s0, iso_forest0, features0 = anomaly_detector.detect_anomalous_rules(
            rules0_df.copy(), contamination=0.07, n_neighbors=20, cluster_id=0
        )
        
        # Explain anomalies
        feature_importance0 = anomaly_detector.explain_anomalies(rules_s0, iso_forest0, features0, cluster_id=0)
        
        # Display results
        anomaly_detector.display_anomaly_results(rules_s0, feature_importance0, cluster_id=0)
        
        # Extract top features for visualization
        if feature_importance0 is not None and len(feature_importance0) >= 2:
            top_features = feature_importance0.nlargest(2, 'importance')['feature'].values
            x_metric, y_metric = top_features[0], top_features[1]
            print(f"Automatically selected features: x_metric={x_metric}, y_metric={y_metric}")
            anomaly_detector.visualize_anomaly_methods(rules_s0, x_metric=x_metric, y_metric=y_metric, cluster_id=0)
        else:
            print("Not enough features in feature_importance or no anomalies detected. Using default metrics.")
            anomaly_detector.visualize_anomaly_methods(rules_s0, x_metric='support', y_metric='lift', cluster_id=0)
        
        # Find business rules
        general_business_rules0 = anomaly_detector.find_business_rules(rules_s0, cluster_id=0, use_initial_filters=True)
        business_rules0 = anomaly_detector.find_business_rules(rules_s0, feature_importance0, cluster_id=0, use_initial_filters=True)
        
        # Extract items analysis
        items_analysis_dict0 = anomaly_detector.extract_items_from_rules(rules_s0)
        for method, df in items_analysis_dict0.items():
            print(f"\nItems Contributing to Anomalous Rules ({method}):")
            print(df.head(10))
        
        # ANOMALY DETECTION FOR CLUSTER 1
        print("\n" + "="*50)
        print("ANOMALY DETECTION FOR CLUSTER 1") 
        print("="*50)
        
        rules1_df, features1_scaled, scaler1 = anomaly_detector.preprocess_rules(rules_cluster1.copy())
        print(f"Cluster 1: Processed {len(rules1_df)} rules")
        
        anomaly_detector.analyze_iforest_contamination(rules1_df.copy(), 1)
        anomaly_detector.analyze_lof_parameters(rules1_df.copy(), 1)
        
        rules_s1, iso_forest1, features1 = anomaly_detector.detect_anomalous_rules(
            rules1_df.copy(), contamination=0.07, n_neighbors=30, cluster_id=1
        )
        feature_importance1 = anomaly_detector.explain_anomalies(rules_s1, iso_forest1, features1, cluster_id=1)
        anomaly_detector.display_anomaly_results(rules_s1, feature_importance1, cluster_id=1)
        
        if feature_importance1 is not None and len(feature_importance1) >= 2:
            top_features = feature_importance1.nlargest(2, 'importance')['feature'].values
            x_metric, y_metric = top_features[0], top_features[1]
            print(f"Automatically selected features: x_metric={x_metric}, y_metric={y_metric}")
            anomaly_detector.visualize_anomaly_methods(rules_s1, x_metric=x_metric, y_metric=y_metric, cluster_id=1)
        else:
            print("Not enough features in feature_importance or no anomalies detected. Using default metrics.")
            anomaly_detector.visualize_anomaly_methods(rules_s1, x_metric='support', y_metric='lift', cluster_id=1)
        
        general_business_rules1 = anomaly_detector.find_business_rules(rules_s1, cluster_id=1, use_initial_filters=True)
        business_rules1 = anomaly_detector.find_business_rules(rules_s1, feature_importance1, cluster_id=1, use_initial_filters=True)
        
        items_analysis_dict1 = anomaly_detector.extract_items_from_rules(rules_s1)
        for method, df in items_analysis_dict1.items():
            print(f"\nItems Contributing to Anomalous Rules ({method}):")
            print(df.head(10))
        
        # ANOMALY DETECTION FOR CLUSTER 2
        print("\n" + "="*50)
        print("ANOMALY DETECTION FOR CLUSTER 2")
        print("="*50)
        
        rules2_df, features2_scaled, scaler2 = anomaly_detector.preprocess_rules(rules_cluster2.copy())
        print(f"Cluster 2: Processed {len(rules2_df)} rules")
        
        anomaly_detector.analyze_iforest_contamination(rules2_df.copy(), 2)
        anomaly_detector.analyze_lof_parameters(rules2_df.copy(), 2)
        
        rules_s2, iso_forest2, features2 = anomaly_detector.detect_anomalous_rules(
            rules2_df.copy(), contamination=0.07, n_neighbors=20, cluster_id=2
        )
        feature_importance2 = anomaly_detector.explain_anomalies(rules_s2, iso_forest2, features2, cluster_id=2)
        anomaly_detector.display_anomaly_results(rules_s2, feature_importance2, cluster_id=2)
        
        if feature_importance2 is not None and len(feature_importance2) >= 2:
            top_features = feature_importance2.nlargest(2, 'importance')['feature'].values
            x_metric, y_metric = top_features[0], top_features[1]
            print(f"Automatically selected features: x_metric={x_metric}, y_metric={y_metric}")
            anomaly_detector.visualize_anomaly_methods(rules_s2, x_metric=x_metric, y_metric=y_metric, cluster_id=2)
        else:
            print("Not enough features in feature_importance or no anomalies detected. Using default metrics.")
            anomaly_detector.visualize_anomaly_methods(rules_s2, x_metric='support', y_metric='lift', cluster_id=2)
        
        # Special analysis for cluster 2 (from original code)
        high_support_rules = rules_s2[rules_s2['support_normalized'] > 5] if 'support_normalized' in rules_s2.columns else pd.DataFrame()
        if not high_support_rules.empty:
            print(f"\nFound {len(high_support_rules)} rules with very high normalized support")
            print(high_support_rules[['rule_str', 'support', 'support_normalized']])
        
        general_business_rules2 = anomaly_detector.find_business_rules(rules_s2, cluster_id=2, use_initial_filters=True)
        business_rules2 = anomaly_detector.find_business_rules(rules_s2, feature_importance2, cluster_id=2, use_initial_filters=True)
        
        if general_business_rules2 is not None:
            print(f"\nGeneral business rules for Cluster 2:")
            print(general_business_rules2)
        if business_rules2 is not None:
            print(f"\nAnomalous business rules for Cluster 2:")
            print(business_rules2)
        
        items_analysis_dict2 = anomaly_detector.extract_items_from_rules(rules_s2)
        for method, df in items_analysis_dict2.items():
            print(f"\nItems Contributing to Anomalous Rules ({method}):")
            print(df.head(10))
        
        # ANOMALY DETECTION FOR CLUSTER 3
        print("\n" + "="*50)
        print("ANOMALY DETECTION FOR CLUSTER 3")
        print("="*50)
        
        rules3_df, features3_scaled, scaler3 = anomaly_detector.preprocess_rules(rules_cluster3.copy())
        print(f"Cluster 3: Processed {len(rules3_df)} rules")
        
        anomaly_detector.analyze_iforest_contamination(rules3_df.copy(), 3)
        # Special LOF parameters for cluster 3 (from original code)
        anomaly_detector.analyze_lof_parameters(rules3_df.copy(), 3, [150, 200, 250, 300, 350])
        
        rules_s3, iso_forest3, features3 = anomaly_detector.detect_anomalous_rules(
            rules3_df.copy(), contamination=0.07, n_neighbors=250, cluster_id=3
        )
        feature_importance3 = anomaly_detector.explain_anomalies(rules_s3, iso_forest3, features3, cluster_id=3)
        anomaly_detector.display_anomaly_results(rules_s3, feature_importance3, cluster_id=3)
        
        if feature_importance3 is not None and len(feature_importance3) >= 2:
            top_features = feature_importance3.nlargest(2, 'importance')['feature'].values
            x_metric, y_metric = top_features[0], top_features[1]
            print(f"Automatically selected features: x_metric={x_metric}, y_metric={y_metric}")
            anomaly_detector.visualize_anomaly_methods(rules_s3, x_metric=x_metric, y_metric=y_metric, cluster_id=3)
        else:
            print("Not enough features in feature_importance or no anomalies detected. Using default metrics.")
            anomaly_detector.visualize_anomaly_methods(rules_s3, x_metric='support', y_metric='lift', cluster_id=3)
        
        general_business_rules3 = anomaly_detector.find_business_rules(rules_s3, cluster_id=3, use_initial_filters=True)
        business_rules3 = anomaly_detector.find_business_rules(rules_s3, feature_importance3, cluster_id=3, use_initial_filters=True)
        
        if general_business_rules3 is not None:
            print(f"\nGeneral business rules for Cluster 3:")
            print(general_business_rules3)
        if business_rules3 is not None:
            print(f"\nAnomalous business rules for Cluster 3:")
            print(business_rules3)
        
        items_analysis_dict3 = anomaly_detector.extract_items_from_rules(rules_s3)
        for method, df in items_analysis_dict3.items():
            print(f"\nItems Contributing to Anomalous Rules ({method}):")
            print(df.head(10))
        
        # ANOMALY DETECTION FOR CLUSTER 4
        print("\n" + "="*50)
        print("ANOMALY DETECTION FOR CLUSTER 4")
        print("="*50)
        
        rules4_df, features4_scaled, scaler4 = anomaly_detector.preprocess_rules(rules_cluster4.copy())
        print(f"Cluster 4: Processed {len(rules4_df)} rules")
        
        anomaly_detector.analyze_iforest_contamination(rules4_df.copy(), 4)
        anomaly_detector.analyze_lof_parameters(rules4_df.copy(), 4)
        
        rules_s4, iso_forest4, features4 = anomaly_detector.detect_anomalous_rules(
            rules4_df.copy(), contamination=0.1, n_neighbors=30, cluster_id=4  # Note: different contamination rate
        )
        feature_importance4 = anomaly_detector.explain_anomalies(rules_s4, iso_forest4, features4, cluster_id=4)
        anomaly_detector.display_anomaly_results(rules_s4, feature_importance4, cluster_id=4)
        
        if feature_importance4 is not None and len(feature_importance4) >= 2:
            top_features = feature_importance4.nlargest(2, 'importance')['feature'].values
            x_metric, y_metric = top_features[0], top_features[1]
            print(f"Automatically selected features: x_metric={x_metric}, y_metric={y_metric}")
            anomaly_detector.visualize_anomaly_methods(rules_s4, x_metric=x_metric, y_metric=y_metric, cluster_id=4)
        else:
            print("Not enough features in feature_importance or no anomalies detected. Using default metrics.")
            anomaly_detector.visualize_anomaly_methods(rules_s4, x_metric='support', y_metric='lift', cluster_id=4)
        
        general_business_rules4 = anomaly_detector.find_business_rules(rules_s4, cluster_id=4, use_initial_filters=True)
        business_rules4 = anomaly_detector.find_business_rules(rules_s4, feature_importance4, cluster_id=4, use_initial_filters=True)
        
        if general_business_rules4 is not None:
            print(f"\nGeneral business rules for Cluster 4:")
            print(general_business_rules4)
        if business_rules4 is not None:
            print(f"\nAnomalous business rules for Cluster 4:")
            print(business_rules4)
        
        items_analysis_dict4 = anomaly_detector.extract_items_from_rules(rules_s4)
        for method, df in items_analysis_dict4.items():
            print(f"\nItems Contributing to Anomalous Rules ({method}):")
            print(df.head(10))
        
        # Save all models and results        
        print("\n" + "="*50)
        print("ANOMALY DETECTION ANALYSIS COMPLETE!")
        print("="*50)
        
        return {
            'anomaly_detector': anomaly_detector,
            'cluster_results': {
                0: {'rules': rules_s0, 'feature_importance': feature_importance0, 'business_rules': business_rules0},
                1: {'rules': rules_s1, 'feature_importance': feature_importance1, 'business_rules': business_rules1},
                2: {'rules': rules_s2, 'feature_importance': feature_importance2, 'business_rules': business_rules2},
                3: {'rules': rules_s3, 'feature_importance': feature_importance3, 'business_rules': business_rules3},
                4: {'rules': rules_s4, 'feature_importance': feature_importance4, 'business_rules': business_rules4}
            }
        }
        
    except Exception as e:
        print(f"Error in anomaly detection workflow: {e}")
        print("Make sure the association rules data exists at 'data/processed/association_rules.csv'")
        return None

if __name__ == "__main__":
    main()