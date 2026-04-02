import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.model_selection import cross_val_score

def train_model(X_train, y_train, X_test, y_test, preprocessor, model_path):
    """Train and save the model."""
    # Create model
    model = RandomForestClassifier(
        random_state=42,
        n_estimators=100,
        class_weight="balanced",
        max_depth=10
    )
    
    # Create pipeline
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    # Train model
    pipe.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = pipe.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipe, model_path)
    
    return pipe, metrics

def load_model(model_path):
    """Load the trained model."""
    return joblib.load(model_path)

def predict_proba(model, X):
    """Get prediction probabilities from the model."""
    return model.predict_proba(X)
