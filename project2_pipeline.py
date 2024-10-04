from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, accuracy_score, roc_auc_score
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from collections import Counter


def check_metrics(X_test, y_test, model):
    # Use the pipeline to make predictions
    y_pred = model.predict(X_test)

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    baccuracy = balanced_accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"\Accuracy of the model: {accuracy:.4f}")
    print(f"Balanced accuracy of the model: {baccuracy:.4f}")
    print(f"ROC-AUC of the model: {roc_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return accuracy

def get_best_pipeline(pipeline_as_is, pipeline_undersampling, pipeline_oversampling, X,y, model):
    
    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

    # Fit the first pipeline
    pipeline_as_is.fit(X_train, y_train)

    print("\n\n*****Testing Dataset as Is") 
    # Print out the MSE, r-squared, and adjusted r-squared values
    # and collect the adjusted r-squared for the first pipeline
    pipeline_as_is_balanced_accuracy = check_metrics(X_test, y_test, pipeline_as_is)
    print(f'\ny_test .info:')
    y_test.info()
    print(f'\ny_test  as_is valuecounts')
    print(y_test.value_counts())

    print("\n\n***** Testing DataSet Undersampling")

    # Fit the second pipeline
    rus = RandomUnderSampler(random_state=1)
    # Fit the data to the model
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
    # Count distinct resampled values
    print(f'\ny_resampled.info:')
    y_resampled.info()
    print(f'\ny_resampled.valuecounts:')
    print(y_resampled.value_counts())

    pipeline_undersampling.fit(X_resampled, y_resampled)

    # Print out the MSE, r-squared, and adjusted r-squared values
    # and collect the adjusted r-squared for the second pipeline
    pipeline_undersample_balanced_accuracy = check_metrics(X_test, y_test, pipeline_undersampling)

    print("\n\n*****Testing DataSet OverSampling")

    # Instantiate the RandomOverSampler instance
    random_oversampler = RandomOverSampler(random_state=1)
    # Fit the data to the model
    X_resampled, y_resampled = random_oversampler.fit_resample(X_train, y_train)
    # Count distinct values
    print(f'\ny_resampled.info:')
    y_resampled.info()
    print(f'\ny_resampled.valuecounts:')
    print(y_resampled.value_counts())

    pipeline_oversampling.fit(X_resampled, y_resampled)
    pipeline_oversampled_balanced_accuracy = check_metrics(X_test, y_test, pipeline_oversampling)

    scores = {"as_is": pipeline_as_is_balanced_accuracy,"undersampled" : pipeline_undersample_balanced_accuracy,"oversampled":pipeline_oversampled_balanced_accuracy} 
    k = Counter(scores)
    high = k.most_common(1) #Fist Highest
    print(f'Dataset with Highest Accuracy for modem {model[0]} : {high[0]}')


def binary_model_generator(X,y,model):
    """
    Creates Pipelines for a single Binary Feature Prediction and identifies the best model
    X: Rest of the DataFrame with all the features minus the Binary feature to predict
    y: Binary Feature to predict in the DataFrame
    """

    print(f'\n******** Starting to run Dataset with  Model: {model[0]}')
    # Create a list of steps for a pipeline that will one hot encode and scale data
    # Each step should be a tuple with a name and a function
    steps = [model] 

    # Determine if OverSampling or Undersampling is necessary Todo:

    # Create a pipeline object to run the data as is
    pipeline_as_is = Pipeline(steps)

    # Create a pipeline object using undersampling data to handle inbalances
    pipeline_undersampling = Pipeline(steps)

    # Create a pipeline object using oversampling data to handle 
    pipeline_oversampling = Pipeline(steps)


        # Get the best pipeline
    pipeline = get_best_pipeline(pipeline_as_is, pipeline_undersampling, pipeline_oversampling, X, y, model)

    # Return the trained model
    return pipeline

if __name__ == "__main__":
    print("This script should not be run directly! Import these functions for use in another file.")

    




