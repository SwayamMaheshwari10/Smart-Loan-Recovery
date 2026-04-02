import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

def plot_recovery_status_distribution(df):
    """
    Plot distribution of recovery status.
    """
    plt.figure(figsize=(8, 6))
    sns.countplot(x=df["Recovery_Status"])
    plt.title("Distribution of Recovery Status")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_payment_history_vs_recovery(df):
    """
    Plot payment history vs recovery status.
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(x=df['Payment_History'], hue=df['Recovery_Status'], 
                  hue_order=["Fully Recovered", "Partially Recovered", "Written Off"])
    plt.title("Payment History vs Recovery Status")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_missed_payments_vs_recovery(df):
    """
    Plot missed payments vs recovery status.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df['Recovery_Status'], y=df['Num_Missed_Payments'], hue=df['Recovery_Status'])
    plt.title("Missed Payments vs Recovery Status")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_income_vs_loan_amount(df):
    """
    Plot monthly income vs loan amount colored by recovery status.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df["Monthly_Income"], y=df["Loan_Amount"], hue=df["Recovery_Status"])
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title("Monthly Income vs Loan Amount")
    plt.tight_layout()
    plt.show()

def plot_customer_segments(df):
    """
    Plot customer segments by loan amount and income.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df["Monthly_Income"], y=df["Loan_Amount"], 
                    hue=df["Customer_Segments"], palette="Set2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title("Customer Segments by Loan & Income")
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df):
    """
    Plot correlation heatmap for numeric features.
    """
    plt.figure(figsize=(12, 10))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels=None):
    """
    Plot confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()

def plot_feature_importance(feature_names, importances, top_n=10):
    """
    Plot feature importance.

    Args:
        feature_names (list): List of feature names
        importances (np.array): Feature importance values
        top_n (int): Number of top features to display
    """
    # Get top features
    top_indices = importances.argsort()[-top_n:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    top_importances = importances[top_indices]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_importances, y=top_features)
    plt.title(f"Top {top_n} Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

def plot_risk_score_distribution(risk_scores):
    """
    Plot distribution of risk scores.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(risk_scores, bins=30, alpha=0.7, edgecolor='black')
    plt.title("Distribution of Risk Scores")
    plt.xlabel("Risk Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

def plot_risk_level_distribution(risk_levels):
    """
    Plot distribution of risk levels.
    """
    plt.figure(figsize=(8, 6))
    risk_levels.value_counts().plot(kind='bar')
    plt.title("Distribution of Risk Levels")
    plt.xlabel("Risk Level")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def generate_classification_report(y_true, y_pred, target_names=None):
    """
    Generate and print classification report.
    """
    report = classification_report(y_true, y_pred, target_names=target_names)
    print("Classification Report:")
    print(report)
    return report

def plot_priority_scores(priority_scores, top_n=20):
    """
    Plot priority scores for top cases.
    """
    top_cases = priority_scores.nlargest(top_n)
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_cases)), top_cases.values)
    plt.yticks(range(len(top_cases)), [f"Case {i+1}" for i in range(len(top_cases))])
    plt.title(f"Top {top_n} Priority Cases")
    plt.xlabel("Priority Score")
    plt.tight_layout()
    plt.show()
