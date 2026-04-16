import streamlit as st
import pandas as pd
import plotly.express as px

from src.model import load_model, predict_proba
from src.data_preprocessing import preprocess_for_model, load_preprocessing_artifacts
from src.recovery_rules import add_risk_features, get_priority_cases, generate_recovery_report
from src.util import get_model_path, get_models_path

# Page configuration
st.set_page_config(
    page_title="Smart Loan Recovery Dashboard",
    page_icon="💰",
    layout="wide"
)

@st.cache_resource
def load_model_and_artifacts():
    """Load the trained model and preprocessing artifacts."""
    try:
        model_path = get_model_path()
        models_dir = get_models_path()
        
        model = load_model(model_path)
        kmeans_model, kmeans_scaler = load_preprocessing_artifacts(models_dir)
        
        return model, kmeans_model, kmeans_scaler
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def process_uploaded_data(df, model, kmeans_model, kmeans_scaler):
    """Process uploaded data and generate predictions."""
    try:
        # Check for required columns
        required_columns = [
            'Monthly_Income', 'Outstanding_Loan_Amount', 'Monthly_EMI', 
            'Collateral_Value', 'Loan_Amount', 'Num_Missed_Payments'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            return None
        
        # Preprocess data using centralized functions
        df_processed, _, _ = preprocess_for_model(df, kmeans_model, kmeans_scaler)
        
        # Prepare features for prediction
        columns_to_drop = ["High_risk_flag", "Customer_Segments", "Segment_name"]
        if "Recovery_Status" in df_processed.columns:
            columns_to_drop.append("Recovery_Status")
        
        X = df_processed.drop(columns=columns_to_drop)
        
        # Generate risk scores
        risk_scores = predict_proba(model, X)[:, 1]
        
        # Add risk features
        df_with_risk = add_risk_features(df_processed.copy(), risk_scores)
        
        return df_with_risk
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None

def main():
    """Main Streamlit app."""
    st.title("💰 Smart Loan Recovery Dashboard")
    st.markdown(
        """
        <style>
        .stApp {font-size: 16px;}
        h1 {font-size: 2.1rem !important;}
        h2, h3 {font-size: 1.35rem !important;}
        .stMetric label, .stMetric div {font-size: 1rem !important;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    # Load model
    with st.spinner("Loading model..."):
        model, kmeans_model, kmeans_scaler = load_model_and_artifacts()
    
    if model is None:
        st.error("Failed to load model. Please run 'python train.py' first.")
        st.stop()
    
    # Sidebar
    st.sidebar.header("📊 Upload Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="Upload a CSV file with loan data for analysis"
    )
    
    if uploaded_file is not None:
        # Load and process data
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Successfully loaded {len(df)} records")
        
        # Process data
        with st.spinner("Processing data and generating predictions..."):
            df_processed = process_uploaded_data(df, model, kmeans_model, kmeans_scaler)
        
        if df_processed is not None:
            # Generate recovery report
            recovery_report = generate_recovery_report(df_processed)
            
            # Create tabs
            tab1, tab2, tab3 = st.tabs(["📈 Metrics", "📊 Charts", "🎯 Priority Cases"])
            
            with tab1:
                st.header("Key Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Cases", f"{recovery_report['total_cases']:,}")
                
                with col2:
                    st.metric("Total Outstanding", f"Rs.{recovery_report['total_outstanding']:,.0f}")
                
                with col3:
                    high_risk_cases = sum(1 for level in df_processed['Risk_Level'] 
                                        if level in ['High', 'Critical'])
                    st.metric("High Risk Cases", f"{high_risk_cases:,}")
                
                with col4:
                    st.metric("High Risk Outstanding", f"Rs.{recovery_report['high_risk_outstanding']:,.0f}")
                
                # Risk distribution
                st.subheader("Risk Level Distribution")
                risk_dist = df_processed['Risk_Level'].value_counts()
                st.dataframe(risk_dist, use_container_width=True)
            
            with tab2:
                st.header("Analysis Charts")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Risk distribution pie chart
                    risk_counts = df_processed['Risk_Level'].value_counts()
                    fig1 = px.pie(
                        values=risk_counts.values,
                        names=risk_counts.index,
                        title="Risk Level Distribution"
                    )
                    fig1.update_layout(font=dict(size=14), title_font=dict(size=20))
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    # Outstanding amount by risk level
                    risk_summary = df_processed.groupby('Risk_Level')['Outstanding_Loan_Amount'].sum().reset_index()
                    fig2 = px.bar(
                        risk_summary,
                        x='Risk_Level',
                        y='Outstanding_Loan_Amount',
                        title="Outstanding Amount by Risk Level"
                    )
                    fig2.update_layout(
                        font=dict(size=14),
                        title_font=dict(size=20),
                        xaxis_title="Risk Level",
                        yaxis_title="Outstanding Amount (Rs.)"
                    )
                    st.plotly_chart(fig2, use_container_width=True)
            
            with tab3:
                st.header("Priority Cases")
                
                # Top priority cases
                priority_cases = get_priority_cases(df_processed, 20)
                top_10_cases = priority_cases.head(10).copy()
                top_10_cases['Case_ID'] = top_10_cases.index.astype(str)
                top_10_cases['Priority_Score'] = top_10_cases['Priority_Score'].round(3)
                
                # Priority chart with clear labels for quick decision-making
                fig3 = px.bar(
                    top_10_cases,
                    x='Priority_Score',
                    y='Case_ID',
                    color='Risk_Level',
                    text='Priority_Score',
                    orientation='h',
                    title="Top 10 Priority Cases"
                )
                fig3.update_traces(textposition='outside')
                fig3.update_layout(
                    font=dict(size=14),
                    title_font=dict(size=20),
                    xaxis_title="Priority Score (Higher = More Urgent)",
                    yaxis_title="Case ID",
                    yaxis=dict(categoryorder='total ascending'),
                    legend_title="Risk Level"
                )
                st.plotly_chart(fig3, use_container_width=True)
                
                # Priority cases table
                st.subheader("Top Priority Cases Details")
                display_cols = ['Outstanding_Loan_Amount', 'Risk_Level', 'Priority_Score', 'Recovery_Action']
                st.dataframe(priority_cases[display_cols], use_container_width=True)
            
            # Download results
            st.header("💾 Download Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv_data = df_processed.to_csv(index=False)
                st.download_button(
                    label="📥 Download Full Results (CSV)",
                    data=csv_data,
                    file_name="loan_recovery_results.csv",
                    mime="text/csv"
                )
            
            with col2:
                priority_csv = get_priority_cases(df_processed, 50).to_csv(index=False)
                st.download_button(
                    label="📥 Download Priority Cases (CSV)",
                    data=priority_csv,
                    file_name="priority_cases.csv",
                    mime="text/csv"
                )
    
    else:
        # Welcome message
        st.header("👋 Welcome to Smart Loan Recovery Dashboard")
        
        st.markdown("""
        This dashboard helps you analyze loan recovery data and generate actionable insights.
        
        ### Features:
        - **Risk Assessment**: Automatically classify loans by risk level
        - **Priority Ranking**: Identify high-priority cases for recovery
        - **Recovery Strategies**: Generate tailored recovery actions
        - **Visual Analytics**: Interactive charts and visualizations
        
        ### How to use:
        1. Upload a CSV file with loan data using the sidebar
        2. View key metrics and risk distribution
        3. Analyze priority cases and recovery strategies
        4. Download results for further analysis
        
        ### Required CSV columns:
        **Essential columns:**
        - `Monthly_Income` - Monthly income of borrower
        - `Outstanding_Loan_Amount` - Current outstanding loan amount
        - `Monthly_EMI` - Monthly EMI payment
        - `Collateral_Value` - Value of collateral
        - `Loan_Amount` - Original loan amount
        - `Num_Missed_Payments` - Number of missed payments
        """)
        
        # Sample data format
        st.subheader("📋 Sample Data Format")
        sample_data = {
            'Monthly_Income': [50000, 75000, 40000],
            'Outstanding_Loan_Amount': [150000, 200000, 120000],
            'Monthly_EMI': [15000, 20000, 12000],
            'Collateral_Value': [200000, 300000, 150000],
            'Loan_Amount': [200000, 300000, 150000],
            'Num_Missed_Payments': [2, 0, 5]
        }
        st.dataframe(pd.DataFrame(sample_data), use_container_width=True)

if __name__ == "__main__":
    main()
