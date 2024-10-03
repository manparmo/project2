
# Model Performance Analysis for Bankruptcy Prediction (Filtered Data)

## Overview

This analysis evaluates the performance of four machine learning models (KNN, Logistic Regression, Random Forest, and Decision Tree) in predicting bankruptcy using a filtered dataset with fewer features. The models were tested under three data treatments: "As Is", "Oversampled", and "Undersampled" to handle the imbalanced nature of the dataset.

## Models Evaluated
1. **K-Nearest Neighbors (KNN)**
2. **Logistic Regression**
3. **Random Forest**
4. **Decision Tree**

The evaluation metrics include Balanced Accuracy, Precision, Recall, and F1-Score for each class (0: Non-bankrupt, 1: Bankrupt).

## Summary of Model Performance

| Model              | Data Treatment | Balanced Accuracy | Precision Class 0 | Recall Class 0 | F1-score Class 0 | Precision Class 1 | Recall Class 1 | F1-score Class 1 |
|--------------------|----------------|-------------------|-------------------|----------------|------------------|-------------------|----------------|------------------|
| KNN                | As Is          | 0.9660            | 0.97              | 0.99           | 0.98             | 0.41              | 0.13           | 0.19             |
| KNN                | Oversampled    | 0.9097            | 0.99              | 0.92           | 0.95             | 0.21              | 0.65           | 0.32             |
| KNN                | Undersampled   | 0.8381            | 0.99              | 0.84           | 0.91             | 0.15              | 0.85           | 0.25             |
| Logistic Regression| As Is          | 0.9683            | 0.97              | 1.00           | 0.98             | 0.67              | 0.04           | 0.07             |
| Logistic Regression| Oversampled    | 0.8592            | 0.99              | 0.86           | 0.92             | 0.17              | 0.84           | 0.28             |
| Logistic Regression| Undersampled   | 0.8575            | 0.99              | 0.86           | 0.92             | 0.16              | 0.84           | 0.27             |
| Random Forest      | As Is          | 0.9695            | 0.97              | 1.00           | 0.98             | 0.58              | 0.20           | 0.30             |
| Random Forest      | Oversampled    | 0.9689            | 0.98              | 0.99           | 0.98             | 0.54              | 0.27           | 0.36             |
| Random Forest      | Undersampled   | 0.8721            | 0.99              | 0.87           | 0.93             | 0.18              | 0.80           | 0.29             |
| Decision Tree      | As Is          | 0.9501            | 0.98              | 0.97           | 0.97             | 0.25              | 0.27           | 0.26             |
| Decision Tree      | Oversampled    | 0.9572            | 0.98              | 0.98           | 0.98             | 0.31              | 0.27           | 0.29             |
| Decision Tree      | Undersampled   | 0.7935            | 0.99              | 0.79           | 0.88             | 0.11              | 0.80           | 0.20             |

## Analysis by Model

### 1. K-Nearest Neighbors (KNN):
- **Balanced Accuracy**: KNN showed strong performance with the "As Is" dataset (0.9660) but had a notable drop in balanced accuracy with undersampled data (0.8381). Its performance was improved by oversampling (0.9097).
- **Minority Class (Class 1) Detection**: While precision and recall for Class 1 were consistently low in the "As Is" case (Precision: 0.41, Recall: 0.13), both recall and F1-score showed significant improvement with oversampling, reaching 0.65 recall and 0.32 F1-score, indicating better detection of bankrupt companies with increased data.

### 2. Logistic Regression:
- **Balanced Accuracy**: Logistic Regression performed very well with the "As Is" treatment (0.9683) but had a lower balanced accuracy with oversampling and undersampling, both around 0.85.
- **Minority Class Detection**: Although Logistic Regression had a very high precision for Class 0, it showed difficulty in correctly identifying bankrupt companies, as indicated by the low recall (0.04) in the "As Is" scenario. However, with oversampling and undersampling, recall improved significantly to 0.84, improving the F1-score as well.

### 3. Random Forest:
- **Balanced Accuracy**: Random Forest consistently performed the best across all treatments, with the highest balanced accuracy for the "As Is" (0.9695) and very close values with oversampling (0.9689).
- **Minority Class Detection**: Precision and recall for the minority class improved in the "Oversampled" scenario (Precision: 0.54, Recall: 0.27), making Random Forest the most effective model for bankruptcy detection, particularly when working with balanced datasets.

### 4. Decision Tree:
- **Balanced Accuracy**: The Decision Tree model performed well with the "As Is" data (0.9501), but its balanced accuracy was notably lower when using undersampled data (0.7935). Oversampling resulted in improved performance (0.9572).
- **Minority Class Detection**: Decision Tree showed consistently moderate performance across all scenarios for detecting the minority class, with a maximum recall of 0.80 in the undersampled data. However, it still struggled with precision and overall F1-score.

## Key Insights and Conclusion

1. **Overall Performance**:
   - The **Random Forest** model again showed the best overall performance, maintaining high balanced accuracy and good minority class detection across different data treatments. This makes it the most reliable model for this problem, particularly in detecting bankruptcy.

2. **Effect of Feature Reduction**:
   - Reducing the number of features seemed to have minimal impact on Random Forest's balanced accuracy compared to the original dataset. However, other models, particularly **KNN** and **Logistic Regression**, showed significant variation in performance with the filtered data, indicating that they might be more sensitive to the number of features available.
   - **Logistic Regression** and **KNN** showed better recall with oversampling and undersampling, suggesting that feature reduction could help improve focus on key indicators of bankruptcy.

3. **Balanced Accuracy**:
   - The **"As Is"** scenario gave the best balanced accuracy for most models, particularly for **Random Forest** and **Logistic Regression**.
   - **Oversampling** generally helped the models perform better in terms of detecting minority classes without sacrificing much accuracy, particularly for Random Forest and KNN.

4. **Minority Class Detection (Bankrupt Companies)**:
   - **Random Forest** and **KNN** performed significantly better in detecting bankrupt companies when oversampling was used, which increased recall significantly.
   - **Decision Tree** and **Logistic Regression** showed improvement with oversampling, but still had lower overall F1-scores for the minority class.

## Recommendations:
- For bankruptcy detection using a filtered dataset, **Random Forest with Oversampling** remains the recommended approach due to its superior balance between high accuracy and effective detection of the minority class.
- Reducing features did not significantly affect Random Forest, which indicates its robustness. However, for simpler models like **Logistic Regression**, retaining key features and using balanced datasets is important.
- Further improvements could include hyperparameter optimization of **Random Forest** and **Decision Tree** models to enhance recall and precision for the minority class.
"""

