import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
import joblib
import os

def load_data(data_path):
    """Load loan recovery data from CSV file."""
    return pd.read_csv(data_path)

def add_features(df):
    """Add engineered features to the dataframe."""
    # Remove ID columns if they exist
    if 'Borrower_ID' in df.columns:
        df = df.drop(columns=['Borrower_ID'])
    if 'Loan_ID' in df.columns:
        df = df.drop(columns=['Loan_ID'])
    
    # Feature engineering
    df["Debt_to_Income"] = df["Outstanding_Loan_Amount"] / df["Monthly_Income"]
    df["EMI_to_Income"] = df["Monthly_EMI"] / df["Monthly_Income"]
    df["Collateral_Coverage"] = df["Collateral_Value"] / df["Loan_Amount"]
    df["Loan_Utilization"] = df["Outstanding_Loan_Amount"] / df["Loan_Amount"]
    
    return df

def create_customer_segments(df, kmeans_model=None, scaler=None):
    """Create customer segments using K-means clustering."""
    features_for_clustering = [
        'Monthly_Income', 'Loan_Amount', 'Outstanding_Loan_Amount',
        'Debt_to_Income', 'EMI_to_Income', 'Num_Missed_Payments'
    ]
    
    if kmeans_model is None or scaler is None:
        # Train new models
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[features_for_clustering])
        
        n_clusters = min(4, len(df))
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df["Customer_Segments"] = kmeans_model.fit_predict(scaled_features)
    else:
        # Use pre-trained models
        scaled_features = scaler.transform(df[features_for_clustering])
        df["Customer_Segments"] = kmeans_model.predict(scaled_features)
    
    # Map segments to names
    segment_names = {
        0: 'Moderate Income, High Loan Burden',
        1: 'High Income, Low Default Risk',
        2: 'Moderate Income, Medium Risk',
        3: 'High Loan, Higher Default Risk'
    }
    
    actual_segments = df['Customer_Segments'].unique()
    segment_mapping = {seg: segment_names.get(seg, f'Segment {seg}') for seg in actual_segments}
    df['Segment_name'] = df['Customer_Segments'].map(segment_mapping)
    
    # Create high risk flag
    df['High_risk_flag'] = df['Segment_name'].apply(
        lambda x: 1 if x in ["Moderate Income, High Loan Burden", "High Loan, Higher Default Risk"] else 0
    )
    
    return df, kmeans_model, scaler

def preprocess_for_model(df, kmeans_model=None, scaler=None):
    """Complete preprocessing pipeline for model training or prediction."""
    # Add engineered features
    df = add_features(df.copy())
    
    # Create customer segments
    df, kmeans_model, scaler = create_customer_segments(df, kmeans_model, scaler)
    
    return df, kmeans_model, scaler

def prepare_features_target(df):
    """Prepare features (X) and target (y) for model training."""
    X = df.drop(columns=["Recovery_Status", "High_risk_flag", "Customer_Segments", "Segment_name"])
    y = df['High_risk_flag']
    return X, y

def get_feature_types(X):
    """Get numeric and categorical feature columns."""
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    return numeric_features, categorical_features

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train and test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def create_preprocessor(numeric_features, categorical_features):
    """Create preprocessing pipeline for numeric and categorical features."""
    return ColumnTransformer([
        ("encoder", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("scaler", StandardScaler(), numeric_features)
    ])

def save_preprocessing_artifacts(kmeans_model, scaler, models_dir):
    """Save KMeans model and scaler for later use."""
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(kmeans_model, os.path.join(models_dir, 'kmeans_model.pkl'))
    joblib.dump(scaler, os.path.join(models_dir, 'kmeans_scaler.pkl'))

def load_preprocessing_artifacts(models_dir):
    """Load KMeans model and scaler."""
    kmeans_path = os.path.join(models_dir, 'kmeans_model.pkl')
    scaler_path = os.path.join(models_dir, 'kmeans_scaler.pkl')
    
    if os.path.exists(kmeans_path) and os.path.exists(scaler_path):
        return joblib.load(kmeans_path), joblib.load(scaler_path)
    return None, None
