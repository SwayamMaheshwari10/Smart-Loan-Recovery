# Smart Loan Recovery System

A simple machine learning project to help prioritize loan recovery cases.

## 🚀 Features

- **Automated Risk Assessment**: ML-powered risk classification (Critical, High, Medium, Low)
- **Priority Ranking**: Intelligent prioritization of recovery cases
- **Recovery Strategies**: Tailored action recommendations for each case
- **Interactive Dashboard**: Clean Streamlit interface with tabs
- **Modular Design**: Clean, modular codebase

## 📁 Project Structure

```
Smart Loan Recovery/
├── train.py                       # Model training script
├── requirements.txt               # Python dependencies
├── data/
│   └── loan-recovery.csv          # Input dataset
├── models/                        # Saved model artifacts
├── reports/                       # Generated CSV outputs
├── notebooks/
│   └── Smart_loan_recovery.ipynb
├── src/
│   ├── data_preprocessing.py      # Feature engineering + segmentation
│   ├── model.py                   # Train/load/predict model
│   ├── recovery_rules.py          # Risk labels + actions + reporting              
│   └── util.py                    # Common path helpers
└── app.py                         # Streamlit dashboard
```

## 🛠️ Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model**:
   ```bash
   python train.py
   ```

3. **Launch the dashboard**:
   ```bash
   streamlit run app.py
   ```

## 📊 Usage

### Training
The `train.py` script:
- Loads data from `data/loan-recovery.csv`
- Applies centralized preprocessing
- Trains a Random Forest classifier
- Saves model and KMeans artifacts to `models/`

### Dashboard
The Streamlit app provides:
- **Metrics Tab**: Key performance indicators
- **Charts Tab**: Risk distribution and outstanding amounts
- **Priority Cases Tab**: Top priority cases with recovery actions
- **Download**: Export results as CSV



## 📋 Required Data Format

**Essential columns:**
- `Monthly_Income` - Monthly income of borrower
- `Outstanding_Loan_Amount` - Current outstanding loan amount
- `Monthly_EMI` - Monthly EMI payment
- `Collateral_Value` - Value of collateral
- `Loan_Amount` - Original loan amount
- `Num_Missed_Payments` - Number of missed payments

**Optional columns:**
- `Age`, `Gender`, `Employment_Type`, `Payment_History`, etc.


## 📈 Model Performance

- **Algorithm**: Random Forest with balanced class weights
- **Target**: Binary high-risk flag derived from K-Means customer segments
- **Features**: 22 engineered features (debt-to-income, EMI-to-income ratios, etc.)
- **Segmentation**: K-Means clustering (4 clusters) for customer profiling
- **Risk Levels**: 4-level rule-based classification applied to model probability scores (Critical, High, Medium, Low)

| Metric | Score |
|---|---|
| Accuracy | 92.00% |
| Precision | 86.84% |
| F1-Score | 89.19% |
| Training Samples | 400 |
| Test Samples | 100 |


## 🔍 Example Workflow

1. **Prepare data**: Ensure CSV has required columns
2. **Train model**: Run `python train.py`
3. **Launch dashboard**: Run `streamlit run app.py`
4. **Upload data**: Use sidebar to upload your CSV
5. **Analyze results**: View metrics, charts, and priority cases
6. **Export results**: Download analysis results
