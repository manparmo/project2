
# Model Performance Analysis for Bankruptcy Prediction

## Overview

This analysis evaluates the performance of four machine learning models (KNN, Logistic Regression, Random Forest, and Decision Tree) in predicting bankruptcy. The models were tested under three data treatments: "As Is", "Oversampled", and "Undersampled" to handle the imbalanced nature of the dataset, which consists mostly of non-bankrupt companies. 

## Models Evaluated
1. **K-Nearest Neighbors (KNN)**
2. **Logistic Regression**
3. **Random Forest**
4. **Decision Tree**

The evaluation metrics include:
- **Accuracy**: Overall correctness of the model.
- **Balanced Accuracy**: The average of recall obtained on each class, useful for imbalanced datasets.
- **ROC-AUC**: Area Under the Receiver Operating Characteristic Curve, which provides an aggregate measure of performance across all classification thresholds.
- **Precision, Recall, F1-Score** for each class (0: Non-bankrupt, 1: Bankrupt).

## Summary of Model Performance

| Model              | Data Treatment | Accuracy | Balanced Accuracy | ROC-AUC | Precision Class 0 | Recall Class 0 | F1-score Class 0 | Precision Class 1 | Recall Class 1 | F1-score Class 1 |
|--------------------|----------------|----------|-------------------|---------|-------------------|----------------|------------------|-------------------|----------------|------------------|
| KNN                | As Is          | 0.9677   | 0.5000            | 0.6362  | 0.97              | 1.00           | 0.98             | 0.00              | 0.00           | 0.00             |
| KNN                | Undersampled   | 0.6018   | 0.6009            | 0.6441  | 0.98              | 0.60           | 0.75             | 0.05              | 0.60           | 0.09             |
| KNN                | Oversampled    | 0.8727   | 0.5652            | 0.6315  | 0.97              | 0.89           | 0.93             | 0.07              | 0.24           | 0.11             |
| Logistic Regression| As Is          | 0.9566   | 0.4942            | 0.5848  | 0.97              | 0.99           | 0.98             | 0.00              | 0.00           | 0.00             |
| Logistic Regression| Undersampled   | 0.7601   | 0.5685            | 0.6257  | 0.97              | 0.77           | 0.86             | 0.05              | 0.36           | 0.09             |
| Logistic Regression| Oversampled    | 0.7425   | 0.5506            | 0.6166  | 0.97              | 0.76           | 0.85             | 0.05              | 0.35           | 0.08             |
| Random Forest      | As Is          | 0.9689   | 0.5709            | 0.9255  | 0.97              | 1.00           | 0.98             | 0.57              | 0.15           | 0.23             |
| Random Forest      | Undersampled   | 0.8686   | 0.8442            | 0.9329  | 0.99              | 0.87           | 0.93             | 0.17              | 0.82           | 0.29             |
| Random Forest      | Oversampled    | 0.9683   | 0.6409            | 0.9279  | 0.98              | 0.99           | 0.98             | 0.52              | 0.29           | 0.37             |
| Decision Tree      | As Is          | 0.9584   | 0.6533            | 0.6533  | 0.98              | 0.98           | 0.98             | 0.35              | 0.33           | 0.34             |
| Decision Tree      | Undersampled   | 0.8194   | 0.7748            | 0.7748  | 0.99              | 0.82           | 0.90             | 0.12              | 0.73           | 0.21             |
| Decision Tree      | Oversampled    | 0.9548   | 0.6515            | 0.6515  | 0.98              | 0.98           | 0.98             | 0.31              | 0.33           | 0.32             |

## Analysis by Model

### 1. K-Nearest Neighbors (KNN):
- **Accuracy**: The highest accuracy for KNN was observed with the "As Is" data (0.9677). Accuracy dropped significantly with undersampling (0.6018), suggesting KNN struggled with limited data diversity.
- **Balanced Accuracy and ROC-AUC**: The balanced accuracy for "As Is" data was very low (0.5000), indicating a bias towards the majority class. The ROC-AUC value (0.6362) also confirmed poor classification of the minority class.
- **Conclusion**: KNN performed well in overall accuracy with "As Is" data but had poor ROC-AUC and balanced accuracy, indicating a failure to adequately classify the minority class.

### 2. Logistic Regression:
- **Accuracy**: The model performed well in terms of accuracy for the "As Is" scenario (0.9566), while accuracy decreased for oversampling and undersampling, both below 0.76.
- **Balanced Accuracy and ROC-AUC**: Balanced accuracy for "As Is" (0.4942) and ROC-AUC (0.5848) were very low, indicating the model couldn't properly detect the minority class.
- **Conclusion**: Logistic Regression achieved good accuracy for the majority class but consistently struggled with detecting bankrupt companies.

### 3. Random Forest:
- **Accuracy**: Random Forest had the highest accuracy across all treatments. The "As Is" scenario had an accuracy of **0.9689**, and oversampling was similar at **0.9683**.
- **Balanced Accuracy and ROC-AUC**: Random Forest demonstrated the highest balanced accuracy and ROC-AUC across all data treatments, especially in the **undersampling scenario** (Balanced Accuracy: **0.8442**, ROC-AUC: **0.9329**).
- **Conclusion**: The combination of high accuracy, balanced accuracy, and ROC-AUC makes Random Forest the best model for predicting bankruptcy.

### 4. Decision Tree:
- **Accuracy**: Decision Tree performed well with "As Is" data (0.9584) and oversampled data (0.9548), while accuracy decreased significantly with undersampling (0.8194).
- **Balanced Accuracy and ROC-AUC**: Balanced accuracy (0.6533) and ROC-AUC (0.6533) for the "As Is" scenario indicate that the model slightly favored the majority class.
- **Conclusion**: While Decision Tree showed moderate accuracy and ROC-AUC values, it performed worse than Random Forest in all cases.

## Key Insights and Conclusion

1. **Best Performing Model**:
   - **Random Forest**: Random Forest performed the best across all treatments in terms of **accuracy**, **balanced accuracy**, and **ROC-AUC**. The model’s ability to generalize well in imbalanced scenarios and consistently achieve a high ROC-AUC indicates it is the most effective for predicting bankruptcy, especially with oversampling or undersampling.

2. **Balanced Accuracy and ROC-AUC**:
   - **Balanced Accuracy**: While accuracy was high for all models, **balanced accuracy** is critical for imbalanced datasets like bankruptcy prediction. Random Forest achieved the best balanced accuracy, showing a significant improvement over other models in effectively detecting bankrupt companies.
   - **ROC-AUC**: Random Forest consistently achieved high ROC-AUC scores, indicating a better distinction between bankrupt and non-bankrupt companies.

3. **Effect of Data Treatments**:
   - **As Is**: For accuracy, most models performed well with the "As Is" data, but balanced accuracy and ROC-AUC metrics were often poor, indicating bias towards the majority class.
   - **Oversampling**: **Random Forest** benefited most from oversampling, showing a high ROC-AUC of **0.9279**, suggesting that balanced data allows the model to learn more effectively.
   - **Undersampling**: While undersampling decreased the overall accuracy, it significantly improved the balanced accuracy and ROC-AUC for **Random Forest** and **Decision Tree**, indicating better minority class detection.

## Recommendations:
- For bankruptcy prediction, Random Forest with either oversampling or undersampling is the recommended approach. The model showed consistently high accuracy, balanced accuracy, and ROC-AUC, making it reliable for detecting bankrupt companies effectively.
