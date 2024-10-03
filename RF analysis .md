# Model Comparison: Random Forest - Filtered vs. Non-Filtered Dataset With and Without Sampling

## Overview

This project aims to evaluate the performance of the **Random Forest** model for bankruptcy prediction using filtered and non-filtered datasets, with and without **over and under sampling**. The primary objective is to understand how feature selection and sampling impact the accuracy and recall of predicting bankrupt companies. Below is a comparison table summarizing the key metrics from different scenarios.

## Comparison Table for Random Forest: Filtered vs. Non-Filtered Dataset With and Without Sampling

| Metric                        | **Non-Filtered Without Sampling** | **Filtered With Sampling** | **Non-Filtered With Sampling** |
|-------------------------------|-----------------------------------|----------------------------|--------------------------------|
| **Accuracy**                  | 0.9526                            | 0.9409                     | 0.9526                         |
| **Precision (Class 0)**       | 0.99                              | 0.99                       | 0.99                           |
| **Precision (Class 1)**       | 0.35                              | 0.30                       | 0.35                           |
| **Recall (Class 0)**          | 0.97                              | 0.95                       | 0.97                           |
| **Recall (Class 1)**          | 0.56                              | 0.65                       | 0.56                           |
| **F1-Score (Class 0)**        | 0.98                              | 0.97                       | 0.98                           |
| **F1-Score (Class 1)**        | 0.43                              | 0.42                       | 0.43                           |
| **Macro Avg Precision**       | 0.67                              | 0.65                       | 0.67                           |
| **Macro Avg Recall**          | 0.76                              | 0.80                       | 0.76                           |
| **Macro Avg F1-Score**        | 0.70                              | 0.69                       | 0.70                           |
| **Weighted Avg Precision**    | 0.96                              | 0.97                       | 0.96                           |
| **Weighted Avg Recall**       | 0.95                              | 0.94                       | 0.95                           |
| **Weighted Avg F1-Score**     | 0.96                              | 0.95                       | 0.96                           |
| **Confusion Matrix (TN, FP, FN, TP)** | [1912, 68, 29, 37]          | [1882, 98, 23, 43]         | [1912, 68, 29, 37]             |

## Summary and Comparison

### 1. Accuracy
- The **non-filtered dataset without sampling** achieved the highest accuracy at **95.26%**, indicating a better fit compared to using sampling.
- Using sampling resulted in slightly lower accuracy due to the intentional balancing of classes, which aims to improve minority class predictions.

### 2. Minority Class (Class 1 - Bankrupt)
- **Recall** for Class 1 improved from `0.56` (without sampling) to `0.65` (filtered with sampling), suggesting better detection of bankruptcies.
- **Precision** decreased with sampling from `0.35` to `0.30`, indicating an increase in false positives.

### 3. Majority Class (Class 0 - Non-Bankrupt)
- The performance for the majority class remained consistently high across all scenarios.
- **Recall** slightly decreased from `0.97` (without sampling) to `0.95` (filtered with sampling), due to the effect of oversampling the minority class.

### 4. Effect of Filtering and Sampling
- The **filtered dataset with sampling** improved the ability to detect bankruptcies (higher recall) while maintaining an acceptable overall accuracy.
- **Undersampling and oversampling** were applied:
  - **Oversampling** increased the number of bankrupt cases (Class 1) to ensure better learning for the minority class.
  - **Undersampling** reduced the number of non-bankrupt cases (Class 0) to help balance the data.

## Conclusion

- If the goal is to **maximize recall for bankruptcies** (Class 1), using the **filtered dataset with oversampling and undersampling** is recommended.
- If **overall accuracy** and minimizing false positives are more critical, using the **non-filtered dataset without sampling** yields better performance.

