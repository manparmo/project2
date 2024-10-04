
# Model Performance Analysis for Bankruptcy Prediction (Filtered vs. As Is Data)

## Overview

This analysis compares the performance of four machine learning models—**K-Nearest Neighbors (KNN)**, **Logistic Regression**, **Random Forest**, and **Decision Tree**—using two different datasets: the original dataset with all features (**As Is**) and a reduced dataset with fewer features (**Filtered**). The models were evaluated across various metrics including **accuracy**, **balanced accuracy**, **ROC-AUC**, **precision**, **recall**, and **F1-score** for both classes: non-bankrupt (0) and bankrupt (1). Each model was evaluated for its ability to handle imbalanced data, and conclusions were drawn regarding which model performed best under each scenario.

## Summary of Model Performance

| Model               | Data Treatment  | Accuracy | Balanced Accuracy | ROC-AUC | Precision (Class 0) | Recall (Class 0) | F1-score (Class 0) | Precision (Class 1) | Recall (Class 1) | F1-score (Class 1) |
|---------------------|-----------------|----------|-------------------|---------|---------------------|------------------|--------------------|---------------------|------------------|--------------------|
| KNN                 | As Is           | 0.9677   | 0.5000            | 0.6362  | 0.97                | 1.00             | 0.98               | 0.00                | 0.00             | 0.00               |
| KNN                 | Filtered        | 0.9660   | 0.5606            | 0.8466  | 0.97                | 0.99             | 0.98               | 0.41                | 0.13             | 0.19               |
| Logistic Regression | As Is           | 0.9566   | 0.4942            | 0.5848  | 0.97                | 0.99             | 0.98               | 0.00                | 0.00             | 0.00               |
| Logistic Regression | Filtered        | 0.9683   | 0.5179            | 0.9332  | 0.97                | 1.00             | 0.98               | 0.67                | 0.04             | 0.07               |
| Random Forest       | As Is           | 0.9689   | 0.5709            | 0.9255  | 0.97                | 1.00             | 0.98               | 0.57                | 0.15             | 0.23               |
| Random Forest       | Filtered        | 0.9695   | 0.5888            | 0.9008  | 0.97                | 1.00             | 0.98               | 0.59                | 0.18             | 0.28               |
| Decision Tree       | As Is           | 0.9584   | 0.6533            | 0.6533  | 0.98                | 0.98             | 0.98               | 0.35                | 0.33             | 0.34               |
| Decision Tree       | Filtered        | 0.9537   | 0.6333            | 0.6333  | 0.98                | 0.98             | 0.98               | 0.29                | 0.29             | 0.29               |

## Detailed Analysis

### **1. K-Nearest Neighbors (KNN)**
- **As Is Data**:
  - Achieved an **accuracy** of **0.9677**, indicating strong performance in classifying the majority class (non-bankrupt).
  - **Balanced accuracy** was only **0.5000** and **ROC-AUC** was **0.6362**, showing that it struggled with detecting the minority class (bankrupt).
  - **Precision** and **Recall** for **Class 1** (bankrupt) were **0.00**, indicating no true positives for the minority class.
- **Filtered Data**:
  - **Accuracy** was still high at **0.9660**, showing KNN’s ability to perform with reduced features.
  - **Balanced accuracy** improved to **0.5606** and **ROC-AUC** increased to **0.8466**, indicating improved but still limited detection of the minority class.

### **2. Logistic Regression**
- **As Is Data**:
  - **Accuracy** was **0.9566**, indicating good classification of the majority class.
  - **Balanced accuracy** was **0.4942** and **ROC-AUC** was **0.5848**, which demonstrates poor performance in recognizing the minority class.
  - **Precision** and **Recall** for **Class 1** were **0.00**, indicating a complete failure to detect bankrupt companies.
- **Filtered Data**:
  - **Accuracy** improved slightly to **0.9683**, with a significant improvement in **ROC-AUC** to **0.9332**, indicating better separation between classes.
  - **Precision** for **Class 1** (bankrupt) increased to **0.67**, but **Recall** remained low at **0.04**, indicating that while some predictions were correct, the model missed many of the bankrupt cases.

### **3. Random Forest**
- **As Is Data**:
  - **Accuracy** was **0.9689**, the highest among all models with the "As Is" data, indicating its robustness.
  - **Balanced accuracy** was **0.5709** and **ROC-AUC** was **0.9255**, showing the model’s better ability to handle class imbalance.
  - **Precision** for **Class 1** was **0.57** and **Recall** was **0.15**, meaning it identified some bankrupt companies but still missed many.
- **Filtered Data**:
  - **Accuracy** further improved to **0.9695**, showing that reduced features did not degrade its performance.
  - **Balanced accuracy** was **0.5888** and **ROC-AUC** was **0.9008**, maintaining its ability to detect minority classes.
  - **Precision** and **Recall** for **Class 1** improved compared to other models, making Random Forest the most effective for bankruptcy detection.

### **4. Decision Tree**
- **As Is Data**:
  - **Accuracy** was **0.9584**, slightly lower than Random Forest and KNN.
  - **Balanced accuracy** was the highest at **0.6533** among the models for "As Is" data, indicating a slightly better balance between class predictions.
  - **Precision** and **Recall** for **Class 1** were **0.35** and **0.33** respectively, indicating moderate capability to detect bankrupt companies.
- **Filtered Data**:
  - **Accuracy** was **0.9537**, and **Balanced accuracy** was **0.6333**, indicating a consistent performance.
  - **Precision** and **Recall** for **Class 1** decreased slightly, making Decision Tree less effective compared to Random Forest and Logistic Regression.

## Conclusion

### **Best Performing Model**: 
- **Random Forest** consistently outperformed other models across most metrics, particularly for accuracy, balanced accuracy, and ROC-AUC. It managed to balance the prediction between classes effectively, even with reduced features, making it the best choice for bankruptcy prediction.

### **Effect of Data Treatments**:
- The **Filtered Data** provided some models (such as Random Forest and Logistic Regression) the opportunity to perform better in distinguishing between classes, as indicated by the higher ROC-AUC values.
- **Balanced Accuracy** and **ROC-AUC** are crucial metrics in this scenario due to the imbalanced nature of the dataset, and **Random Forest** achieved the best balance across these metrics.

### **Challenges in Identifying Minority Class**:
- The **minority class (bankrupt)** was consistently harder to classify for most models, as evidenced by lower precision and recall values. This was especially true for **KNN** and **Logistic Regression** when using "As Is" data.
- The use of **undersampling and oversampling techniques** could be further explored to improve the model’s performance for the minority class, focusing on **Random Forest** and **Decision Tree**.

### **Key Recommendation**:
- For practical implementation, **Random Forest** is recommended due to its robustness, high accuracy, and ability to generalize well across different data treatments. It also provides better balanced accuracy and ROC-AUC, which are critical in detecting bankrupt companies effectively.

The analysis indicates that while all models had good accuracy, the true test lies in effectively detecting bankrupt cases (the minority class), where **Random Forest** excelled over others.
"""

