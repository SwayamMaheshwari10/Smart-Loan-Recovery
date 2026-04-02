import os

def get_project_root():
    """Get the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_data_path(filename="loan-recovery.csv"):
    """Get the path to data file."""
    project_root = get_project_root()
    return os.path.join(project_root, "data", filename)

def get_models_path():
    """Get the path to models directory."""
    project_root = get_project_root()
    return os.path.join(project_root, "models")

def get_model_path(model_name="loan_recovery_model.pkl"):
    """Get the path to saved model file."""
    models_path = get_models_path()
    return os.path.join(models_path, model_name)
