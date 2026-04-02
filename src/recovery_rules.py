import pandas as pd
import numpy as np

def classify_risk(score):
    """
    Classify risk level based on risk score.
    
    Args:
        score (float): Risk score between 0 and 1
        
    Returns:
        str: Risk level classification
    """
    if score >= 0.8:
        return "Critical"
    elif score >= 0.6:
        return "High"
    elif score >= 0.4:
        return "Medium"
    else:
        return "Low"

def generate_recovery_action(row):
    """
    Generate one clear recovery action per case.
    
    Args:
        row (pd.Series): Row containing risk level and other features
        
    Returns:
        str: Single recovery action recommendation
    """
    if row["Risk_Level"] == "Critical":
        return "Escalate immediately to collections/legal"

    if row["Risk_Level"] == "High":
        return "Offer structured EMI restructuring"

    if row["Risk_Level"] == "Medium":
        if row["Num_Missed_Payments"] >= 3:
            return "Start focused follow-up with a fixed repayment plan"
        return "Send reminder and confirm monthly payment commitment"

    return "Continue regular monitoring"

def calculate_priority_score(risk_score, outstanding_amount, max_outstanding):
    """
    Calculate priority score for loan recovery.
    
    Args:
        risk_score (float): Risk score from model
        outstanding_amount (float): Outstanding loan amount
        max_outstanding (float): Maximum outstanding amount in dataset
        
    Returns:
        float: Priority score
    """
    return risk_score * 0.6 + (outstanding_amount / max_outstanding) * 0.4

def add_risk_features(df, risk_scores):
    """
    Add risk-related features to dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        risk_scores (np.array): Risk scores from model
        
    Returns:
        pd.DataFrame: Dataframe with added risk features
    """
    df["Risk_score"] = risk_scores
    df["Risk_Level"] = df["Risk_score"].apply(classify_risk)
    df["Recovery_Action"] = df.apply(generate_recovery_action, axis=1)
    
    # Calculate priority score
    max_outstanding = df["Outstanding_Loan_Amount"].max()
    df["Priority_Score"] = df.apply(
        lambda row: calculate_priority_score(
            row["Risk_score"], 
            row["Outstanding_Loan_Amount"], 
            max_outstanding
        ), axis=1
    )
    
    return df

def get_priority_cases(df, top_n=50):
    """
    Get top priority cases for recovery.
    
    Args:
        df (pd.DataFrame): Dataframe with priority scores
        top_n (int): Number of top cases to return
        
    Returns:
        pd.DataFrame: Top priority cases
    """
    return df.sort_values("Priority_Score", ascending=False).head(top_n)

def get_risk_summary(df):
    """
    Get summary statistics by risk level.
    
    Args:
        df (pd.DataFrame): Dataframe with risk levels
        
    Returns:
        pd.DataFrame: Summary statistics by risk level
    """
    summary = df.groupby('Risk_Level').agg({
        'Outstanding_Loan_Amount': ['count', 'sum', 'mean'],
        'Risk_score': 'mean',
        'Priority_Score': 'mean'
    }).round(2)
    
    return summary

def get_recovery_strategy_summary(df):
    """
    Get summary of recovery strategies.
    
    Args:
        df (pd.DataFrame): Dataframe with recovery actions
        
    Returns:
        pd.DataFrame: Summary of recovery strategies
    """
    # Count occurrences of each strategy component
    all_strategies = []
    for action in df['Recovery_Action']:
        strategies = action.split('; ')
        all_strategies.extend(strategies)
    
    strategy_counts = pd.Series(all_strategies).value_counts()
    return strategy_counts

def generate_recovery_report(df):
    """
    Generate comprehensive recovery report.
    
    Args:
        df (pd.DataFrame): Dataframe with all risk and recovery features
        
    Returns:
        dict: Dictionary containing various report components
    """
    report = {
        'total_cases': len(df),
        'risk_distribution': df['Risk_Level'].value_counts().to_dict(),
        'total_outstanding': df['Outstanding_Loan_Amount'].sum(),
        'high_risk_outstanding': df[df['Risk_Level'].isin(['High', 'Critical'])]['Outstanding_Loan_Amount'].sum(),
        'priority_cases': get_priority_cases(df, 10),
        'risk_summary': get_risk_summary(df),
        'strategy_summary': get_recovery_strategy_summary(df)
    }
    
    return report
