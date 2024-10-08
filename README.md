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
    - Due to the weak correlation of individual features, complex models like **Random Forest** were used.
  
      ### Key Features in the Filtered Dataset:
1. Net Income to Stockholder's Equity
2. Debt Ratio %
3. Current Liability to Assets
4. Borrowing Dependency
5. Retained Earnings to Total Assets
6. Total Asset Turnover
7. Operating Gross Margin
8. Net Income to Total Assets
9. Cash Flow to Total Assets
10. Interest Coverage Ratio (Interest expense to EBIT)

## Data Preprocessing
Since the dataset was highly imbalanced, with far fewer bankrupt companies compared to non-bankrupt ones, we applied several techniques:
- **Oversampling** to balance the classes.
- **Undersampling** to reduce the majority class.
- **Feature Selection** to focus on the most relevant features.

## Models Implemented
We implemented the following models:
- **K-Nearest Neighbors (KNN)**
- **Logistic Regression**
- **Random Forest**
- **Decision Tree**

Each model was trained and evaluated on both the full dataset and the filtered dataset.

## Model Performance

| Model                | Data Treatment | Accuracy | Balanced Accuracy | ROC-AUC | Precision (Class 0) | Recall (Class 0) | F1-score (Class 0) | Precision (Class 1) | Recall (Class 1) | F1-score (Class 1) |
|----------------------|----------------|----------|-------------------|---------|---------------------|------------------|--------------------|---------------------|------------------|--------------------|
| **KNN**              | As Is          | 0.9677   | 0.5000            | 0.6362  | 0.97                | 1.00             | 0.98               | 0.00                | 0.00             | 0.00               |
| **KNN**              | Filtered       | 0.9660   | 0.5606            | 0.8466  | 0.97                | 0.99             | 0.98               | 0.41                | 0.13             | 0.19               |
| **Logistic Regression** | As Is       | 0.9566   | 0.4942            | 0.5848  | 0.97                | 0.99             | 0.98               | 0.00                | 0.00             | 0.00               |
| **Logistic Regression** | Filtered    | 0.9683   | 0.5179            | 0.9332  | 0.97                | 1.00             | 0.98               | 0.67                | 0.04             | 0.07               |
| **Random Forest**     | As Is          | 0.9689   | 0.5709            | 0.9255  | 0.97                | 1.00             | 0.98               | 0.57                | 0.15             | 0.23               |
| **Random Forest**     | Filtered       | 0.9695   | 0.5888            | 0.9008  | 0.97                | 1.00             | 0.98               | 0.59                | 0.18             | 0.28               |
| **Decision Tree**     | As Is          | 0.9584   | 0.6533            | 0.6533  | 0.98                | 0.98             | 0.98               | 0.35                | 0.33             | 0.34               |
| **Decision Tree**     | Filtered       | 0.9537   | 0.6333            | 0.6333  | 0.98                | 0.98             | 0.98               | 0.29                | 0.29             | 0.29               |


### Models Overview
**Decistion Tree**
This model was used to analyze and predict the binary data results.

#### Results:
1. **Accuracy**: The model achieved an accuracy score of approximately **0.9501**, indicating that the model correctly classified around 95.01% of the companies.
2. **Classification Report**:
- Precision:
For class 0: Precision = 0.98 (98% of predictions for class 0 were correct).
For class 1: Precision = 0.33 (Only 33% of predictions for class 1 were correct).
- Recall:
For class 0: Recall = 0.97 (97% of actual class 0 instances were correctly predicted).
For class 1: Recall = 0.38 (Only 38% of actual class 1 instances were correctly predicted).
- F1-Score:
For class 0: F1-score = 0.98.
For class 1: F1-score = 0.35.
- Insights:
The model performs very well on class 0 but struggles with class 1, which has lower precision, recall, and F1-score. This could indicate a class imbalance problem, where the model is biased toward the majority class (0).

## Summary of Model Performance for Filtered Dataset

| Model              | Data Treatment | Accuracy | Balanced Accuracy | ROC-AUC | Precision Class 0 | Recall Class 0 | F1-score Class 0 | Precision Class 1 | Recall Class 1 | F1-score Class 1 |
|--------------------|----------------|----------|-------------------|---------|-------------------|----------------|------------------|-------------------|----------------|------------------|
| **KNN**            | As Is          | 0.9660   | 0.5606            | 0.8466  | 0.97              | 0.99           | 0.98             | 0.41              | 0.13           | 0.19             |
| **KNN**            | Undersampled   | 0.8381   | 0.8461            | 0.9106  | 0.99              | 0.84           | 0.91             | 0.15              | 0.85           | 0.25             |
| **KNN**            | Oversampled    | 0.9097   | 0.7864            | 0.8402  | 0.99              | 0.92           | 0.95             | 0.21              | 0.65           | 0.32             |
| **Logistic Regression** | As Is     | 0.9683   | 0.5179            | 0.9332  | 0.97              | 1.00           | 0.98             | 0.67              | 0.04           | 0.07             |
| **Logistic Regression** | Undersampled | 0.8575 | 0.8473            | 0.9296  | 0.99              | 0.86           | 0.92             | 0.16              | 0.84           | 0.27             |
| **Logistic Regression** | Oversampled | 0.8592 | 0.8482            | 0.9333  | 0.99              | 0.86           | 0.92             | 0.17              | 0.84           | 0.28             |
| **Random Forest**   | As Is          | 0.9695   | 0.5888            | 0.9008  | 0.97              | 1.00           | 0.98             | 0.59              | 0.18           | 0.28             |
| **Random Forest**   | Undersampled   | 0.8545   | 0.8282            | 0.9183  | 0.99              | 0.86           | 0.92             | 0.16              | 0.80           | 0.26             |
| **Random Forest**   | Oversampled    | 0.9689   | 0.6324            | 0.8979  | 0.98              | 0.99           | 0.98             | 0.54              | 0.27           | 0.36             |
| **Decision Tree**   | As Is          | 0.9537   | 0.6333            | 0.6333  | 0.98              | 0.98           | 0.98             | 0.29              | 0.29           | 0.29             |
| **Decision Tree**   | Undersampled   | 0.7871   | 0.7933            | 0.7933  | 0.99              | 0.79           | 0.88             | 0.11              | 0.80           | 0.20             |
| **Decision Tree**   | Oversampled    | 0.9554   | 0.6079            | 0.6079  | 0.97              | 0.98           | 0.98             | 0.28              | 0.24           | 0.25             |

---


# Analysis by Model

### 1. **K-Nearest Neighbors (KNN)**:
- **Accuracy**: The highest accuracy was achieved with "As Is" data (0.9660), but accuracy dropped with undersampling (0.8381).
- **Balanced Accuracy & ROC-AUC**: Undersampling improved balanced accuracy (0.8461) and ROC-AUC (0.9106), showing benefits for class balance.
- **Conclusion**: KNN performed well with "As Is" data for overall accuracy, but balanced accuracy and ROC-AUC were improved with undersampling, making it more reliable for detecting bankrupt companies.

### 2. **Logistic Regression**:
- **Accuracy**: The highest accuracy was achieved with "As Is" data (0.9683).
- **Balanced Accuracy & ROC-AUC**: With undersampling and oversampling, balanced accuracy improved (up to 0.8482), while the ROC-AUC reached up to 0.9333.
- **Conclusion**: Logistic Regression performs best with balanced data treatments, particularly with undersampling, achieving high ROC-AUC and balanced accuracy.

### 3. **Random Forest**:
- **Accuracy**: Random Forest consistently achieved the highest accuracy across all treatments, with 0.9695 in "As Is" data.
- **Balanced Accuracy & ROC-AUC**: Random Forest achieved the best balanced accuracy (0.8282 with undersampling) and high ROC-AUC, indicating strong class distinction.
- **Conclusion**: Random Forest is the most effective model for handling imbalanced datasets, excelling in accuracy, balanced accuracy, and ROC-AUC.

### 4. **Decision Tree**:
- **Accuracy**: Decision Tree performed moderately well with "As Is" and oversampled data (accuracies of 0.9537 and 0.9554).
- **Balanced Accuracy & ROC-AUC**: Decision Tree's balanced accuracy was lower than Random Forest, and it was more prone to overfitting.
- **Conclusion**: Decision Tree showed moderate performance but lagged behind Random Forest, especially in balanced accuracy and generalization.

1. **Dependencies**: 
- Python 3.x
- Libraries: `pandas`, `numpy`, `matplotlib`, `scikit-learn`
2. **Data Preparation**:
- Clean the dataset and ensure all column names are correctly formatted.

## Conclusion

The analysis suggests that predicting bankruptcy is a complex task involving multiple factors. No single financial metric is a strong predictor; thus, utilizing a combination of features with advanced machine learning models is necessary to build a reliable prediction model.
