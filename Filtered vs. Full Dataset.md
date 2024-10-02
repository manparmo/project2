# Model Comparison: Filtered vs. Full Dataset for KNN, Random Forest, and Logistic Regression

## Overview

This project evaluates the performance of three machine learning models — **KNN (K-Nearest Neighbors)**, **Random Forest**, and **Logistic Regression** — on a bankruptcy prediction dataset. The models are compared using both **filtered** and **full datasets** to determine the impact of feature selection on classification performance. The focus is on identifying the best model for predicting bankruptcies, with metrics such as **Accuracy**, **Precision**, **Recall**, and **F1-Score** used for comparison.

## Comparison Table

| Metric                        | **KNN (Filtered)** | **KNN (Full)**  | **Random Forest (Filtered)** | **Random Forest (Full)** | **Logistic Regression (Filtered)** | **Logistic Regression (Full)** |
|-------------------------------|--------------------|-----------------|------------------------------|--------------------------|------------------------------------|--------------------------------|
| **Accuracy**                  | 0.96               | 0.9619          | 0.96                         | 0.9677                   | 0.85                               | 0.7463                         |
| **Precision (Class 1)**       | 0.50               | 0.33            | 0.61                         | 0.82                     | 0.19                               | 0.05                           |
| **Recall (Class 1)**          | 0.18               | 0.02            | 0.18                         | 0.18                     | 0.91                               | 0.39                           |
| **F1-Score (Class 1)**        | 0.26               | 0.04            | 0.28                         | 0.29                     | 0.31                               | 0.09                           |
| **Confusion Matrix**          | [[1954, 14], [64, 14]] | [[1311, 2], [50, 1]] | [[1959, 9], [64, 14]]  | [[1311, 2], [42, 9]] | [[1666, 302], [7, 71]]             | [[1001, 319], [27, 17]]        |

## Key Observations

### 1. KNN (Filtered vs. Full Dataset)
- **Accuracy** was similar for both datasets, indicating that the feature selection had minimal impact on the overall performance.
- **Precision** for identifying bankrupt companies improved significantly with the filtered dataset (`0.50` vs. `0.33`), suggesting a reduction in false positives.
- **Recall** improved from `0.02` (full) to `0.18` (filtered), showing that more bankrupt companies were correctly identified when using the filtered dataset.
- **F1-Score** also improved from `0.04` to `0.26`, reflecting a better balance between precision and recall for the filtered dataset.

### 2. Random Forest (Filtered vs. Full Dataset)
- **Accuracy** was high for both datasets, with the full dataset slightly outperforming the filtered dataset (`0.9677` vs. `0.96`).
- **Precision** improved for the full dataset (`0.82` vs. `0.61`), indicating fewer false positives when using all features.
- **Recall** remained the same for both datasets (`0.18`), showing that the ability to detect bankruptcies was unaffected by feature selection.
- The **F1-Score** remained similar (`0.28` vs. `0.29`), suggesting comparable overall effectiveness.

### 3. Logistic Regression (Filtered vs. Full Dataset)
- **Accuracy** increased significantly with the filtered dataset (`0.85` vs. `0.7463`), showing that feature selection had a positive effect on prediction accuracy.
- **Precision** for predicting bankruptcies improved from `0.05` (full) to `0.19` (filtered), indicating fewer false positives with the filtered dataset.
- **Recall** increased dramatically from `0.39` to `0.91`, showing that the filtered dataset allowed for better identification of actual bankruptcies.
- **F1-Score** improved from `0.09` to `0.31`, reflecting a more balanced performance between precision and recall.

## Conclusion

### Filtered vs. Full Dataset
- **Feature selection** generally led to improved performance for **Logistic Regression** and **KNN**, particularly for detecting bankrupt companies.
- **Random Forest** showed consistent performance across both datasets, with slightly better precision for the full dataset.

### Best Model for Identifying Bankruptcies
- **Logistic Regression (Filtered Dataset)** had the **highest recall** (`0.91`) and a significantly improved **F1-score** (`0.31`), making it suitable for minimizing false negatives and ensuring that most bankruptcies are identified.
- **Random Forest (Full Dataset)** had the **highest precision** (`0.82`), which makes it the best choice for avoiding false positives when predicting bankruptcies.

### Recommendations
- If minimizing **false negatives** is the priority (e.g., to capture as many bankruptcies as possible), **Logistic Regression with the filtered dataset** is recommended.
- If avoiding **false positives** is more important (to reduce unnecessary alerts about bankruptcies), consider using **Random Forest with the full dataset**.


