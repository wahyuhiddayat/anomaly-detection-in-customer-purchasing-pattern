import sys
import warnings
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')
sys.path.append('src')

from anomaly_detection import AnomalyDetection

# Configure Streamlit page
st.set_page_config(
    page_title="Anomaly Detection in Customer Purchasing Patterns",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: white !important;
        color: black !important;
    }
    
    /* Main header */
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 600;
        color: #1f2937;
        text-align: center;
        margin-bottom: 1rem;
        line-height: 1.2;
    }
    
    .sub-header {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        font-weight: 400;
        color: #6b7280;
        text-align: center;
        margin-bottom: 3rem;
        letter-spacing: 0.5px;
    }
    
    /* Section headers */
    .section-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: #374151;
        margin: 2rem 0 1rem 0;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 0.5rem;
    }
    
    /* Metric cards */
    .metric-container {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        padding: 0 20px;
        background-color: transparent;
        border-radius: 6px;
        color: #64748b;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e2e8f0;
        color: #475569;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #3b82f6;
        color: white;
        box-shadow: 0 2px 4px 0 rgba(59, 130, 246, 0.25);
    }
    
    /* Card styling */
    .analysis-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }
    
    /* Data quality indicators */
    .status-success {
        color: #10b981;
        font-weight: 500;
    }
    
    .status-warning {
        color: #f59e0b;
        font-weight: 500;
    }
    
    .status-error {
        color: #ef4444;
        font-weight: 500;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_processed_data():
    """Load all processed data files."""
    try:
        # Load processed data
        cleaned_data = pd.read_csv('data/processed/cleaned_retail_data.csv')
        customer_segments = pd.read_csv('data/processed/customer_segments.csv')
        association_rules = pd.read_csv('data/processed/association_rules.csv')
        cluster_summary = pd.read_csv('data/processed/cluster_summary.csv')
        
        # Convert date column - handle PyArrow compatibility
        cleaned_data['InvoiceDate'] = pd.to_datetime(cleaned_data['InvoiceDate']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Convert frozensets in association rules
        association_rules['antecedents'] = association_rules['antecedents'].apply(eval)
        association_rules['consequents'] = association_rules['consequents'].apply(eval)
        
        return cleaned_data, customer_segments, association_rules, cluster_summary
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

def main():
    """Main Streamlit application."""
    
    # Main header
    st.markdown('<h1 class="main-header">Anomaly Detection in Customer Purchasing Patterns</h1>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading data..."):
        cleaned_data, customer_segments, association_rules, cluster_summary = load_processed_data()
    
    if cleaned_data is None:
        st.error("Could not load required data files. Please ensure all data files are in data/processed/")
        return
    
    # Main navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Executive Overview", 
        "🎯 Customer Segmentation", 
        "🔍 Association Mining", 
        "⚠️ Anomaly Detection"
    ])
    
    # Tab 1: Executive Overview
    with tab1:
        st.markdown('<h2 class="section-header">Executive Summary</h2>', unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_customers = len(customer_segments) if customer_segments is not None else 0
            st.metric("Total Customers", f"{total_customers:,}")
            
        with col2:
            total_transactions = len(cleaned_data) if cleaned_data is not None else 0
            st.metric("Total Transactions", f"{total_transactions:,}")
            
        with col3:
            total_rules = len(association_rules) if association_rules is not None else 0
            st.metric("Association Rules", f"{total_rules:,}")
            
        with col4:
            if cleaned_data is not None:
                total_revenue = cleaned_data['TotalPrice'].sum()
                st.metric("Total Revenue", f"${total_revenue:,.2f}")
        
        # Key Performance Indicators
        if cleaned_data is not None and customer_segments is not None:
            st.markdown('<h3 class="section-header">Key Performance Indicators</h3>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Revenue Analysis**")
                daily_revenue = cleaned_data.groupby(pd.to_datetime(cleaned_data['InvoiceDate']).dt.date)['TotalPrice'].sum()
                fig = px.line(x=daily_revenue.index, y=daily_revenue.values, 
                             title="Daily Revenue Trend",
                             color_discrete_sequence=['#3b82f6'])
                fig.update_layout(
                    xaxis_title="Date", 
                    yaxis_title="Revenue ($)",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_family="Inter",
                    title_font_size=16,
                    title_font_color='#374151'
                )
                fig.update_traces(line_width=3)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**Customer Distribution**")
                if 'Cluster' in customer_segments.columns:
                    cluster_counts = customer_segments['Cluster'].value_counts().sort_index()
                    fig = px.pie(values=cluster_counts.values, names=[f"Cluster {i}" for i in cluster_counts.index],
                               title="Customer Distribution by Cluster")
                    st.plotly_chart(fig, use_container_width=True)
        
        # Data Summary
        st.markdown('<h3 class="section-header">Data Summary</h3>', unsafe_allow_html=True)
        if cleaned_data is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Transaction Data:**")
                st.dataframe(cleaned_data.describe(), use_container_width=True)
            with col2:
                if customer_segments is not None:
                    st.write("**Customer Segments:**")
                    st.dataframe(customer_segments.describe(), use_container_width=True)
    
    # Tab 2: Customer Segmentation
    with tab2:
        st.markdown('<h2 class="section-header">Customer Segmentation Analysis</h2>', unsafe_allow_html=True)
        
        if customer_segments is not None:
            # Cluster overview
            st.markdown("**Cluster Performance Metrics**")
            
            if cluster_summary is not None:
                # Display cluster summary
                st.dataframe(cluster_summary, use_container_width=True)
            
            # Cluster analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**RFM Analysis by Cluster**")
                if all(col in customer_segments.columns for col in ['Cluster', 'Recency', 'Frequency', 'Monetary']):
                    rfm_by_cluster = customer_segments.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
                    
                    fig = make_subplots(
                        rows=1, cols=3,
                        subplot_titles=('Recency', 'Frequency', 'Monetary'),
                        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
                    )
                    
                    clusters = rfm_by_cluster.index
                    fig.add_trace(go.Bar(x=clusters, y=rfm_by_cluster['Recency'], name='Recency'), row=1, col=1)
                    fig.add_trace(go.Bar(x=clusters, y=rfm_by_cluster['Frequency'], name='Frequency'), row=1, col=2)
                    fig.add_trace(go.Bar(x=clusters, y=rfm_by_cluster['Monetary'], name='Monetary'), row=1, col=3)
                    
                    fig.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**Cluster Characteristics**")
                if 'Cluster' in customer_segments.columns:
                    cluster_selected = st.selectbox("Select Cluster:", sorted(customer_segments['Cluster'].unique()))
                    
                    cluster_data = customer_segments[customer_segments['Cluster'] == cluster_selected]
                    
                    # Cluster metrics
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Customers", len(cluster_data))
                    with col_b:
                        if 'Monetary' in cluster_data.columns:
                            avg_value = cluster_data['Monetary'].mean()
                            st.metric("Avg Value", f"${avg_value:.2f}")
                    with col_c:
                        if 'Frequency' in cluster_data.columns:
                            avg_freq = cluster_data['Frequency'].mean()
                            st.metric("Avg Frequency", f"{avg_freq:.1f}")
                    
                    # Cluster details
                    st.write("**Cluster Details:**")
                    st.dataframe(cluster_data.head(), use_container_width=True)
            
            # Cluster visualization
            st.markdown('<h3 class="section-header">Cluster Visualization</h3>', unsafe_allow_html=True)
            if all(col in customer_segments.columns for col in ['Recency', 'Frequency', 'Monetary', 'Cluster']):
                viz_option = st.selectbox("Select Visualization:", 
                                        ["2D Scatter (Frequency vs Monetary)", "3D Scatter (RFM)", "Parallel Coordinates"])
                
                if viz_option == "2D Scatter (Frequency vs Monetary)":
                    fig = px.scatter(customer_segments, x='Frequency', y='Monetary', color='Cluster',
                                   title="Customer Clusters: Frequency vs Monetary",
                                   hover_data=['Recency'])
                    st.plotly_chart(fig, use_container_width=True)
                
                elif viz_option == "3D Scatter (RFM)":
                    fig = px.scatter_3d(customer_segments, x='Recency', y='Frequency', z='Monetary', 
                                      color='Cluster', title="3D RFM Analysis")
                    st.plotly_chart(fig, use_container_width=True)
                
                elif viz_option == "Parallel Coordinates":
                    fig = px.parallel_coordinates(customer_segments, 
                                                color='Cluster',
                                                dimensions=['Recency', 'Frequency', 'Monetary'],
                                                title="Parallel Coordinates Plot")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Customer segmentation data not available. Please run the segmentation algorithm first.")
    
    # Tab 3: Association Rules
    with tab3:
        st.markdown('<h2 class="section-header">Association Rule Mining</h2>', unsafe_allow_html=True)
        
        if association_rules is not None:
            # Rules overview
            st.markdown("**Rules Performance Summary**")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rules", len(association_rules))
            with col2:
                avg_confidence = association_rules['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.3f}")
            with col3:
                avg_lift = association_rules['lift'].mean()
                st.metric("Avg Lift", f"{avg_lift:.2f}")
            with col4:
                avg_support = association_rules['support'].mean()
                st.metric("Avg Support", f"{avg_support:.3f}")
            
            # Filter rules
            st.markdown('<h3 class="section-header">Rule Explorer & Filtering</h3>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                cluster_filter = st.multiselect("Filter by Cluster:", 
                                              options=sorted(association_rules['Cluster'].unique()),
                                              default=sorted(association_rules['Cluster'].unique()))
                
                min_confidence = st.slider("Minimum Confidence:", 0.0, 1.0, 0.5, 0.01)
                
            with col2:
                min_lift = st.slider("Minimum Lift:", 1.0, association_rules['lift'].max(), 1.5, 0.1)
                min_support = st.slider("Minimum Support:", 0.0, association_rules['support'].max(), 0.01, 0.001)
            
            # Apply filters
            filtered_rules = association_rules[
                (association_rules['Cluster'].isin(cluster_filter)) &
                (association_rules['confidence'] >= min_confidence) &
                (association_rules['lift'] >= min_lift) &
                (association_rules['support'] >= min_support)
            ].copy()
            
            st.write(f"**Showing {len(filtered_rules)} rules (filtered from {len(association_rules)})**")
            
            # Rules table
            if len(filtered_rules) > 0:
                # Create readable rule strings
                filtered_rules['Rule'] = filtered_rules.apply(
                    lambda row: f"{list(row['antecedents'])} → {list(row['consequents'])}", axis=1
                )
                
                display_cols = ['Rule', 'Cluster', 'support', 'confidence', 'lift', 'leverage', 'conviction']
                st.dataframe(filtered_rules[display_cols].round(4), use_container_width=True)
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Support vs Confidence Analysis**")
                    fig = px.scatter(filtered_rules, x='support', y='confidence', 
                                   color='Cluster', size='lift',
                                   hover_data=['leverage', 'conviction'],
                                   title="Association Rules: Support vs Confidence")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("**Lift Distribution Analysis**")
                    fig = px.histogram(filtered_rules, x='lift', color='Cluster',
                                     title="Distribution of Lift Values")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Top rules
                st.markdown("**Top Performing Rules**")
                top_rules = filtered_rules.nlargest(10, 'lift')[['Rule', 'support', 'confidence', 'lift', 'Cluster']]
                st.dataframe(top_rules, use_container_width=True)
            else:
                st.warning("No rules match the current filters. Try adjusting the filter criteria.")
        else:
            st.warning("Association rules data not available. Please run the association rule mining algorithm first.")
    
    # Tab 4: Anomaly Detection
    with tab4:
        st.markdown('<h2 class="section-header">Anomaly Detection</h2>', unsafe_allow_html=True)
        
        if association_rules is not None:
            st.markdown("**Anomaly Detection Analysis**")
            
            # Cluster selection for anomaly analysis
            cluster_for_anomaly = st.selectbox("Select Cluster for Anomaly Analysis:", 
                                             sorted(association_rules['Cluster'].unique()))
            
            cluster_rules = association_rules[association_rules['Cluster'] == cluster_for_anomaly].copy()
            
            if len(cluster_rules) > 0:
                if st.button(f"🔍 Detect Anomalies in Cluster {cluster_for_anomaly}"):
                    with st.spinner("Running anomaly detection..."):
                        try:
                            # Initialize anomaly detector
                            anomaly_detector = AnomalyDetection()
                            
                            # Preprocess rules
                            processed_rules, _, _ = anomaly_detector.preprocess_rules(cluster_rules.copy())
                            
                            # Detect anomalies
                            rules_with_anomalies, iso_forest, features = anomaly_detector.detect_anomalous_rules(
                                processed_rules.copy(), contamination=0.07, n_neighbors=20
                            )
                            
                            # Store results
                            st.session_state[f'anomaly_results_{cluster_for_anomaly}'] = rules_with_anomalies
                        except Exception as e:
                            st.error(f"Error in anomaly detection: {e}")
                
                # Display results if available
                if f'anomaly_results_{cluster_for_anomaly}' in st.session_state:
                    results = st.session_state[f'anomaly_results_{cluster_for_anomaly}']
                    
                    # Anomaly summary
                    iso_anomalies = results['isolation_forest_anomaly'].sum()
                    lof_anomalies = results['lof_anomaly'].sum()
                    combined_anomalies = results['combined_anomaly'].sum()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Rules", len(results))
                    with col2:
                        st.metric("Isolation Forest Anomalies", iso_anomalies)
                    with col3:
                        st.metric("LOF Anomalies", lof_anomalies)
                    with col4:
                        st.metric("Combined Anomalies", combined_anomalies)
                    
                    # Anomaly visualization
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Anomaly Distribution Plot**")
                        fig = px.scatter(results, x='support', y='lift',
                                       color='isolation_forest_anomaly',
                                       title="Anomalies: Support vs Lift",
                                       color_discrete_map={0: 'blue', 1: 'red'},
                                       labels={'isolation_forest_anomaly': 'Anomaly'})
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("**Anomaly Score Analysis**")
                        fig = px.histogram(results, x='anomaly_score',
                                         title="Distribution of Anomaly Scores",
                                         color='isolation_forest_anomaly',
                                         color_discrete_map={0: 'blue', 1: 'red'})
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Anomalous rules table
                    st.markdown("**Detected Anomalous Rules**")
                    anomalous_rules = results[results['isolation_forest_anomaly'] == 1].copy()
                    
                    if len(anomalous_rules) > 0:
                        # Create rule strings
                        anomalous_rules['Rule'] = anomalous_rules.apply(
                            lambda row: f"{list(row['antecedents'])} → {list(row['consequents'])}", axis=1
                        )
                        
                        display_cols = ['Rule', 'support', 'confidence', 'lift', 'anomaly_score']
                        st.dataframe(anomalous_rules[display_cols].round(4), use_container_width=True)
                    else:
                        st.info("No anomalous rules detected with current parameters.")
            else:
                st.warning(f"No rules found for Cluster {cluster_for_anomaly}")
        else:
            st.warning("Anomaly detection requires association rules data.")
            
if __name__ == "__main__":
    main()