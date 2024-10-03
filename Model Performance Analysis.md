
# Model Performance Analysis for Bankruptcy Prediction

## Overview

This analysis evaluates the performance of four machine learning models (KNN, Logistic Regression, Random Forest, and Decision Tree) in predicting bankruptcy. The models were tested under three data treatments: "As Is", "Oversampled", and "Undersampled" to handle the imbalanced nature of the dataset, which consists mostly of non-bankrupt companies.

## Models Evaluated
1. **K-Nearest Neighbors (KNN)**
2. **Logistic Regression**
3. **Random Forest**
4. **Decision Tree**

The evaluation metrics include Balanced Accuracy, Precision, Recall, and F1-Score for each class (0: Non-bankrupt, 1: Bankrupt).

## Summary of Model Performance

| Model              | Data Treatment | Balanced Accuracy | Precision Class 0 | Recall Class 0 | F1-score Class 0 | Precision Class 1 | Recall Class 1 | F1-score Class 1 |
|--------------------|----------------|-------------------|-------------------|----------------|------------------|-------------------|----------------|------------------|
| Decision Tree      | As Is          | 0.9560            | 0.98              | 0.98           | 0.98             | 0.32              | 0.33           | 0.32             |
| Decision Tree      | Oversampled    | 0.9572            | 0.98              | 0.98           | 0.98             | 0.33              | 0.33           | 0.33             |
| Decision Tree      | Undersampled   | 0.8152            | 0.99              | 0.81           | 0.90             | 0.13              | 0.85           | 0.23             |
| KNN                | As Is          | 0.9677            | 0.97              | 1.00           | 0.98             | 0.00              | 0.00           | 0.00             |
| KNN                | Oversampled    | 0.8727            | 0.97              | 0.89           | 0.93             | 0.07              | 0.24           | 0.11             |
| KNN                | Undersampled   | 0.6018            | 0.98              | 0.60           | 0.75             | 0.05              | 0.60           | 0.09             |
| Logistic Regression| As Is          | 0.9566            | 0.97              | 0.99           | 0.98             | 0.00              | 0.00           | 0.00             |
| Logistic Regression| Oversampled    | 0.7425            | 0.97              | 0.76           | 0.85             | 0.05              | 0.35           | 0.08             |
| Logistic Regression| Undersampled   | 0.7601            | 0.97              | 0.77           | 0.86             | 0.05              | 0.36           | 0.09             |
| Random Forest      | As Is          | 0.9666            | 0.97              | 1.00           | 0.98             | 0.43              | 0.11           | 0.17             |
| Random Forest      | Oversampled    | 0.9683            | 0.98              | 0.99           | 0.98             | 0.52              | 0.27           | 0.36             |
| Random Forest      | Undersampled   | 0.8663            | 0.99              | 0.87           | 0.93             | 0.17              | 0.82           | 0.28             |

## Analysis by Model

### 1. Decision Tree:
- **Balanced Accuracy**: The Decision Tree performed consistently well under the "As Is" and "Oversampled" treatments (around 0.9560 - 0.9572), but its performance dropped significantly when the dataset was undersampled (0.8152).
- **Minority Class (Class 1) Detection**: Precision and recall for the minority class (bankrupt companies) were moderate in the "As Is" and "Oversampled" treatments, with an F1-score of around 0.33. However, it showed a significant increase in recall to 0.85 for the undersampled data, indicating that it identified more bankrupt companies, though precision dropped, leading to a lower F1-score (0.23).

### 2. K-Nearest Neighbors (KNN):
- **Balanced Accuracy**: KNN performed very well with the "As Is" data (0.9677) but struggled with undersampled data (0.6018), indicating poor performance with imbalanced datasets unless sufficient data was present.
- **Minority Class Detection**: KNN showed very poor detection capabilities for the minority class across all treatments, with both precision and recall close to zero in most cases. It slightly improved with oversampling (precision of 0.07 and recall of 0.24), but still performed poorly compared to other models.

### 3. Logistic Regression:
- **Balanced Accuracy**: Logistic Regression had high balanced accuracy in the "As Is" scenario (0.9566) but dropped in both "Oversampled" (0.7425) and "Undersampled" (0.7601) data treatments.
- **Minority Class Detection**: Logistic Regression consistently struggled with detecting bankrupt companies, as indicated by near-zero precision and recall across most scenarios. The model’s linear nature may not have been sufficient to capture the complexities of the imbalanced dataset.

### 4. Random Forest:
- **Balanced Accuracy**: Random Forest had the highest and most consistent performance across all data treatments. It achieved the highest balanced accuracy with "Oversampled" data (0.9683).
- **Minority Class Detection**: Random Forest performed significantly better than the other models in identifying bankrupt companies, especially with "Oversampled" data (precision: 0.52, recall: 0.27, F1-score: 0.36). This indicates that Random Forest can effectively deal with imbalanced data and recognize patterns related to the minority class.

## Key Insights and Conclusion

1. **Overall Performance**:
   - The **Random Forest** model stands out as the best performing model overall. It exhibited the highest balanced accuracy and was able to handle the imbalanced nature of the dataset effectively, particularly with the "Oversampled" treatment.
   - The **Decision Tree** also showed competitive performance but struggled more with undersampling compared to Random Forest.

2. **Balanced Accuracy**:
   - **Random Forest** and **KNN** showed the best performance under the "As Is" treatment, but KNN struggled significantly with undersampling, indicating that its performance is sensitive to the quantity of data available.
   - **Logistic Regression** and **Decision Tree** had relatively lower balanced accuracy when undersampled, which indicates a drop in performance when handling fewer samples.

3. **Minority Class (Bankrupt Companies) Detection**:
   - **Random Forest** showed the best ability to identify bankrupt companies, especially when the data was oversampled. This was reflected in higher precision and recall compared to other models.
   - Both **KNN** and **Logistic Regression** showed poor performance for the minority class across all treatments, indicating that they are not suitable for applications where identifying bankrupt companies is critical.

4. **Recommendations**:
   - For a dataset where identifying the minority class (bankrupt companies) is essential, **Random Forest with oversampling** is the recommended model due to its superior balance between overall accuracy and the ability to correctly identify high-risk cases.
   - Future improvements could involve hyperparameter tuning of the Random Forest model and experimenting with other ensemble methods like **Gradient Boosting** or **XGBoost** to further improve minority class detection.
"""
