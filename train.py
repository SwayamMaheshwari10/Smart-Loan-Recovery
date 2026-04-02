import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import (
    load_data, preprocess_for_model, prepare_features_target, 
    get_feature_types, split_data, create_preprocessor,
    save_preprocessing_artifacts
)
from src.model import train_model
from src.recovery_rules import add_risk_features, generate_recovery_report
from src.util import get_data_path, get_model_path, get_models_path

def main():         
    """Main training function."""
    print("Starting Smart Loan Recovery Model Training...")
    print("="*50)  
    
    # Load data
    print("1. Loading data...")
    data_path = get_data_path()
    df = load_data(data_path)
    print(f"   Loaded {len(df)} records")
    
    # Preprocess data
    print("\n2. Preprocessing data...")
    df_processed, kmeans_model, kmeans_scaler = preprocess_for_model(df)
    print(f"   Added engineered features and customer segments")
    
    # Prepare features and target
    print("\n3. Preparing features and target...")
    X, y = prepare_features_target(df_processed)
    print(f"   Features: {X.shape[1]}, Target distribution: {y.value_counts().to_dict()}")
    
    # Get feature types and create preprocessor
    numeric_features, categorical_features = get_feature_types(X)
    preprocessor = create_preprocessor(numeric_features, categorical_features)
    
    # Split data
    print("\n4. Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    
    # Train model
    print("\n5. Training model...")
    model_path = get_model_path()
    pipe, metrics = train_model(X_train, y_train, X_test, y_test, preprocessor, model_path)
    print(f"   Model accuracy: {metrics['accuracy']:.4f}")
    print(f"   Model precision: {metrics['precision']:.4f}")
    print(f"   Model F1-score: {metrics['f1_score']:.4f}")
    
    # Save preprocessing artifacts
    print("\n6. Saving preprocessing artifacts...")
    models_dir = get_models_path()
    save_preprocessing_artifacts(kmeans_model, kmeans_scaler, models_dir)
    print(f"   KMeans model and scaler saved to: {models_dir}")
    
    # Generate risk scores and recovery strategies
    print("\n7. Generating risk scores and recovery strategies...")
    risk_scores = pipe.predict_proba(X)[:, 1]
    df_with_risk = add_risk_features(df_processed.copy(), risk_scores)
    
    # Generate recovery report
    recovery_report = generate_recovery_report(df_with_risk)
    print(f"   Total cases: {recovery_report['total_cases']}")
    print(f"   Total outstanding: Rs.{recovery_report['total_outstanding']:,.0f}")
    print(f"   High risk cases: {sum(1 for level in df_with_risk['Risk_Level'] if level in ['High', 'Critical'])}")
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*50)

if __name__ == "__main__":
    main()
