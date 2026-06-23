# ***Bankruptcy Prediction & Classification Pipeline***

## **Overview**
This project implements a robust Machine Learning pipeline in Python to predict corporate bankruptcy. By analyzing various financial indicators from the provided dataset, the system classifies companies into two categories: Healthy (1) and Bankrupt (2). The project includes comprehensive data preprocessing, class imbalance handling, and the evaluation of multiple classification algorithms using cross-validation.

## **DatasetInput**
-File: Dataset2Use_Assignment1.xlsx  
-Features: Financial indicators (the first 11 columns of the dataset are extracted as features).  
-Target Variable: ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (Indication of inconsistency) (=2) (ν+1).  
-Temporal Data: The column ΕΤΟΣ (Year) is used to track trends over time.

## **Features & Data Processing**
The data pipeline applies several standard data science practices to ensure reliable model training:
-Missing Value Handling: The script automatically scans for NaN values and removes the corresponding rows.  
-Normalization: All feature values are scaled to a [0,1] range using Scikit-Learn's MinMaxScaler.  
-Cross-Validation: Implements StratifiedKFold with 4 splits to maintain class distribution across training and testing sets.  
-Class Imbalance Resolution: The dataset is dynamically down-sampled during the training phase. If healthy companies vastly outnumber bankrupt ones, the healthy class is reduced to maintain a maximum 3:1 ratio (Healthy:Bankrupt). 

## **Modeles Evaluated**
The script tests and compares the following classification algorithms:
-Linear Discriminant Analysis (LDA)  
-Logistic Regression  
-Decision Tree  
-Random Forest  
-K-Nearest Neighbors (KNN)  
-Gaussian Naive Bayes  
-Support Vector Machines (SVM) - RBF Kernel 
-Support Vector Machines (SVM) - Linear Kernel  

## **Evaluation Metrics & Visualizations**
Model performance is evaluated based on:
-Metrics: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.  
-Data Visualizations:
-Bar charts comparing the number of Healthy vs. Bankrupt companies per year.  
-Min/Max/Mean statistical plots for key financial indicators.  
-Confusion Matrices (visualized using Matplotlib) for both Training and Testing sets across all classifiers.  

## **Outputs**
After execution, the evaluation metrics for all classifiers across all folds are aggregated and exported to a CSV file:
-balancedDataOutcomes.csv: Contains detailed results including True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN), and ROC-AUC scores.

## **Requirements**
To run this project, you need Python installed along with the following libraries:  
-pandas  
-numpy  
-matplotlib  
-scikit-learn  
-openpyxl (Required by pandas to read .xlsx files)

## **How to Run**
1) Ensure Dataset2Use_Assignment1.xlsx is placed in the same directory as the Python script.
2) Run the script.
3) The program will display the data visualizations and confusion matrices in sequence. Close each plot window to allow the execution to proceed to the next step.
4) Upon completion, check the root directory for the generated balancedDataOutcomes.csv report.  
