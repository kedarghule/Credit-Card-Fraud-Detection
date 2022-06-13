# Credit-Card-Fraud-Detection

## Problem Statement

Fraud detection find a number of applications in multiple industries. For credit card companies, it is important to be able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase. The problem statement is to detect fraudulent transactions using supervised machine learning techniques on this classification problem.

## Dataset

The dataset used for this project can be found at this [link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

The dataset contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, the data for the original features and more background information about the data cannot be provided. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

## Data Exploration

- The summary of the data is examined.
- The descriptive statistics of the data is examined.
- A check for missing values is performed. Dataset has none.
- A check for class imbalance is performed. As seen below, dataset is highly imbalanced.
  ![image](https://user-images.githubusercontent.com/41315903/173405002-54a83363-9f52-4624-80af-b2b58720756a.png)
- Data distributions are visualized. Results are shown below.

  ![image](https://user-images.githubusercontent.com/41315903/173405168-a3d5296f-e1a4-469a-963a-bf00d17a6a73.png)
  ![image](https://user-images.githubusercontent.com/41315903/173405230-a69806ad-17f2-41f9-b756-df7ce3c16841.png)
- Feature Density plot is examined. The plot can be seen [here](https://github.com/kedarghule/Credit-Card-Fraud-Detection/blob/main/feature_density_plot.png).
  Distribution of Class 0 is plotted in red while distribution of Class 1 is plotted in blue.
    Observations:

      1) Features V4 and V11 have clearly separated distributions for Class 0 and 1.

      2) Following the same trend, features V12, V14, V18 are partially separated for Class 0 and 1.

      3) V1, V2, V3, V10 have a quite distinct profile.

      4) V25, V26, V28 have similar profiles for the two values of Class.

## Data Preprocessing

- **Feature Scaling**: The dataset mentions that all columns with the exception of Time and Amount are scaled. Therefore, we need to scale these two columns as well. We make used of `RobustScaler()` from sklearn.preprocessing as it is less prone to outliers.
- **Splitting the Data**: Data is split into training, validation and test sets.
- **Oversampling using SMOTE**: We perform random oversampling using SMOTE (Synthetic Minority Oversampling Technique) so that we geta a balanced dataset. SMOTE selects a minority class instance at random and finds its k nearest minority class neighbors. 

## Logistic Regression

We start off by using Logistic Regression as our baseline model. To find the optimal hyperparameters `penalty` and `C`, we perform grid search CV with 10-fold cross validation. Our best parameters for the model were observed to be:  {'C': 10, 'penalty': 'l2'}. 
The model is trained again on these parameters and we use the validation and test data to evaluate the model.

**Observations:**

![image](https://user-images.githubusercontent.com/41315903/173409110-7fd56827-ffa6-449a-a68d-493ccbcb20fb.png)

![image](https://user-images.githubusercontent.com/41315903/173409326-38a01000-3359-4512-9e58-57c2a6fb67b0.png)


**The dataset description mentions that given the class imbalance ratio, we it is recommended to evaluate the model using the Area Under the Precision-Recall Curve (AUPRC). Using Logistic Regression, we get AUPRC score of 0.70 on the test data. This can be improved** and hence we look at XGBoost.


## XGBoost Classifier

To increase our average precision-recall score (Area Under the Precision-Recall Curve), XGBoost is considered for better predictive capability. We followed the following steps in this process:
- **Hyperparameter tuning using Hyperopt:**  Hyperopt is a powerful Python library that can optimize a function's value over complex spaces of input i.e., it can optimize a model's accuracy over a space of hyperparameters. It’s a Bayesian optimizer, meaning it is not merely randomly searching or searching a grid, but intelligently learning which combinations of values work well as it goes, and focusing the search there. It is supported by a SMBO methodology adapted to work with different algorithms such as: Tree of Parzen Estimators (TPE), Adaptive Tree of Parzen Estimators (ATPE) and Gaussian Processes (GP).

- **Training the Model:** The model is trained with the optimal parameters we got using Hyperopt for our XGBoost classifier.

**Observations:**

![image](https://user-images.githubusercontent.com/41315903/173432119-23cc9045-e3f2-47b0-889c-3ed4590ebf39.png)

![image](https://user-images.githubusercontent.com/41315903/173432735-7cc393e1-0ee3-49b1-9a5a-ffbd67b5cb33.png)

As we can see, there is a significant increase in our main evaluation metric i.e., the AUPRC score which is 0.82 on the test data for the XGBoost Classifier.

Below is the Precision-Recall Curve for our model:

![image](https://user-images.githubusercontent.com/41315903/173433502-995b0f42-fc6f-4a62-b1b3-da1ccc336ff6.png)
