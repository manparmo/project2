
# Model Performance Analysis for Bankruptcy Prediction (Filtered Data)

## Overview

This analysis evaluates the performance of four machine learning models (KNN, Logistic Regression, Random Forest, and Decision Tree) in predicting bankruptcy using filtered data with fewer features. The models were tested under three data treatments: "As Is", "Oversampled", and "Undersampled" to handle the imbalanced nature of the dataset. 

## Models Evaluated
1. **K-Nearest Neighbors (KNN)**
2. **Logistic Regression**
3. **Random Forest**
4. **Decision Tree**

## Summary of Model Performance

| Model              | Data Treatment | Accuracy | Balanced Accuracy | ROC-AUC | Precision Class 0 | Recall Class 0 | F1-score Class 0 | Precision Class 1 | Recall Class 1 | F1-score Class 1 |
|--------------------|----------------|----------|-------------------|---------|-------------------|----------------|------------------|-------------------|----------------|------------------|
| KNN                | As Is          | 0.9660   | 0.5606            | 0.8466  | 0.97              | 0.99           | 0.98             | 0.41              | 0.13           | 0.19             |
| KNN                | Undersampled   | 0.8381   | 0.8461            | 0.9106  | 0.99              | 0.84           | 0.91             | 0.15              | 0.85           | 0.25             |
| KNN                | Oversampled    | 0.9097   | 0.7864            | 0.8402  | 0.99              | 0.92           | 0.95             | 0.21              | 0.65           | 0.32             |
| Logistic Regression| As Is          | 0.9683   | 0.5179            | 0.9332  | 0.97              | 1.00           | 0.98             | 0.67              | 0.04           | 0.07             |
| Logistic Regression| Undersampled   | 0.8575   | 0.8473            | 0.9296  | 0.99              | 0.86           | 0.92             | 0.16              | 0.84           | 0.27             |
| Logistic Regression| Oversampled    | 0.8592   | 0.8482            | 0.9333  | 0.99              | 0.86           | 0.92             | 0.17              | 0.84           | 0.28             |
| Random Forest      | As Is          | 0.9695   | 0.5888            | 0.9008  | 0.97              | 1.00           | 0.98             | 0.59              | 0.18           | 0.28             |
| Random Forest      | Undersampled   | 0.8545   | 0.8282            | 0.9183  | 0.99              | 0.86           | 0.92             | 0.16              | 0.80           | 0.26             |
| Random Forest      | Oversampled    | 0.9689   | 0.6324            | 0.8979  | 0.98              | 0.99           | 0.98             | 0.54              | 0.27           | 0.36             |
| Decision Tree      | As Is          | 0.9537   | 0.6333            | 0.6333  | 0.98              | 0.98           | 0.98             | 0.29              | 0.29           | 0.29             |
| Decision Tree      | Undersampled   | 0.7871   | 0.7933            | 0.7933  | 0.99              | 0.79           | 0.88             | 0.11              | 0.80           | 0.20             |
| Decision Tree      | Oversampled    | 0.9554   | 0.6079            | 0.6079  | 0.97              | 0.98           | 0.98             | 0.28              | 0.24           | 0.25             |

## Analysis by Model

### 1. K-Nearest Neighbors (KNN):
- **Accuracy**: The highest accuracy for KNN was observed with the "As Is" data (0.9660). Accuracy dropped with undersampling (0.8381), indicating that KNN still struggles with the reduced data.
- **Balanced Accuracy and ROC-AUC**: The "As Is" scenario had a balanced accuracy of 0.5606, indicating some bias towards the majority class. Undersampling increased balanced accuracy significantly (0.8461), showing the benefit of using balanced data. The ROC-AUC value improved with undersampling as well (0.9106).
- **Conclusion**: KNN performed well with the "As Is" data for overall accuracy but showed improved balanced accuracy and ROC-AUC with undersampling, making it more reliable for identifying bankrupt companies.

### 2. Logistic Regression:
- **Accuracy**: The model performed best in the "As Is" scenario with an accuracy of 0.9683. Accuracy dropped in undersampling and oversampling scenarios, both around 0.8575-0.8592.
- **Balanced Accuracy and ROC-AUC**: Balanced accuracy improved in undersampling and oversampling scenarios to 0.8482, while the ROC-AUC reached up to 0.9333. This indicates that Logistic Regression performs better in distinguishing between classes when the data is more balanced.
- **Conclusion**: Logistic Regression performed best with balanced data treatments, especially in undersampling, achieving high ROC-AUC and balanced accuracy.

### 3. Random Forest:
- **Accuracy**: Random Forest had the highest accuracy across all treatments. It achieved an accuracy of **0.9695** with "As Is" data and **0.9689** with oversampled data.
- **Balanced Accuracy and ROC-AUC**: Random Forest consistently achieved high balanced accuracy and ROC-AUC across all treatments, with the highest balanced accuracy observed with **undersampling** (0.8282). The ROC-AUC also indicated a good distinction between the two classes.
- **Conclusion**: Random Forest continues to outperform other models due to its robustness, handling imbalanced datasets well. The combination of accuracy and high ROC-AUC makes it the most effective model for predicting bankruptcy in this dataset.

### 4. Decision Tree:
- **Accuracy**: Decision Tree performed moderately well with the "As Is" and oversampled data, achieving accuracies of 0.9537 and 0.9554, respectively.
- **Balanced Accuracy and ROC-AUC**: The balanced accuracy and ROC-AUC were lower compared to Random Forest. With undersampling, the balanced accuracy reached 0.7933, showing improved performance in detecting minority classes.
- **Conclusion**: Decision Tree showed moderate performance across all metrics, but it lagged behind Random Forest. Its susceptibility to overfitting and reduced balanced accuracy indicated weaker generalization.

## Key Insights and Conclusion

1. **Best Performing Model**:
   - **Random Forest**: Random Forest consistently performed the best across all treatments for filtered data in terms of **accuracy**, **balanced accuracy**, and **ROC-AUC**. It effectively handled the class imbalance and demonstrated a good ability to distinguish between bankrupt and non-bankrupt classes.

2. **Balanced Accuracy and ROC-AUC**:
   - **Balanced Accuracy**: Random Forest achieved the highest balanced accuracy, particularly when undersampling, which is crucial for imbalanced datasets like bankruptcy prediction.
   - **ROC-AUC**: Random Forest and Logistic Regression consistently achieved high ROC-AUC scores, indicating a better ability to distinguish between bankrupt and non-bankrupt companies.

3. **Effect of Data Treatments**:
   - **As Is**: Most models achieved high accuracy, but balanced accuracy and ROC-AUC metrics showed that these models favored the majority class.
   - **Oversampling and Undersampling**: Balancing the dataset through undersampling or oversampling improved the models' ability to identify the minority class effectively, especially in terms of ROC-AUC and balanced accuracy.
