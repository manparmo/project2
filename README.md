#  Company Bankruptcy Prediction

## Project Overview

This project aims to predict the likelihood of bankruptcy for Taiwanese companies using financial data from the Taiwan Economic Journal for the years 1999 to 2009. The dataset consists of various financial ratios and metrics that indicate the financial health of companies. The target variable (`Bankrupt?`) is a binary classification indicating whether a company went bankrupt (1) or not (0).

## Dataset Information

- **Source**: Taiwan Economic Journal
- **Time Period**: 1999 - 2009
- **Number of Instances**: 6819
- **Number of Features**: 95
- **Target Variable**: `Bankrupt?` - Class label indicating bankruptcy status (0 or 1)

## Features

The dataset includes several financial indicators such as:
- **Debt Ratio %**: Represents the ratio of total liabilities to total assets.
- **Operating Gross Margin**: Measures operational efficiency.
- **Net Income to Stockholder's Equity (ROE)**: Indicates profitability relative to shareholder equity.
- **ROA** (Return on Assets): Various forms indicating profitability.
- **Other Financial Ratios**: Current liabilities, borrowing dependency, retained earnings, and more.

## Analysis Summary

### Key Findings:
1. **Correlations**:
    - Most features have weak to moderate correlations with the target variable (`Bankrupt?`).
    - **Debt Ratio %** and **Current Liability to Assets** have positive correlations, suggesting that higher values are linked to increased bankruptcy risk.
    - **Net Income to Total Assets** and **ROA** variants show negative correlations, indicating that higher profitability reduces bankruptcy risk.

2. **Modeling Recommendation**:
    - Due to the weak correlation of individual features, complex models like **Random Forest**, **Gradient Boosting**, are recommended. 

## Logistic Regression Analysis

### Model Overview
- **Logistic Regression** was used as a baseline model to predict bankruptcy.
- It is a linear model used for binary classification, suitable for initial exploration due to its interpretability.

### Results:
1. **Accuracy**: The model achieved an accuracy score of approximately **0.7463**, indicating that the model correctly classified around 74.63% of the companies.
  
2. **Classification Report**:
   - **Precision**: 
     - Class `0` (Non-bankrupt): **0.97** - indicating that the model is highly accurate when predicting companies that are not bankrupt.
     - Class `1` (Bankrupt): **0.05** - indicating that the model struggles to accurately predict bankruptcy cases due to class imbalance.
   - **Recall**: 
     - Class `0`: **0.76** - the model correctly identifies 76% of non-bankrupt companies.
     - Class `1`: **0.39** - the model only identifies 39% of the bankrupt companies, showing a challenge in detecting bankruptcy.
   - **F1-score**: The F1-score for class `1` is **0.09**, indicating a weak performance in predicting bankruptcy cases.

3. **Confusion Matrix**:
- The confusion matrix shows that the model predicted **1001 true negatives** (correctly classified non-bankrupt companies) and **27 true positives** (correctly classified bankrupt companies).
- However, there are **319 false positives** (companies predicted as bankrupt but are not) and **17 false negatives** (companies that went bankrupt but were not predicted to).

### Interpretation and Challenges:
- **Class Imbalance**: The dataset is highly imbalanced, with far fewer bankrupt companies than non-bankrupt ones. This impacts the model's ability to correctly predict the minority class (bankrupt).

1. **Dependencies**: 
- Python 3.x
- Libraries: `pandas`, `numpy`, `matplotlib`, `scikit-learn`
2. **Data Preparation**:
- Clean the dataset and ensure all column names are correctly formatted.

## Conclusion

The analysis suggests that predicting bankruptcy is a complex task involving multiple factors. No single financial metric is a strong predictor; thus, utilizing a combination of features with advanced machine learning models is necessary to build a reliable prediction model.
