# UCIsAdultDataset
 This is a non-scientific study on UCI's Adult Dataset which is donated by Ronny Kohavi and Barry Becker.
 
 
 Source;
 
 https://archive.ics.uci.edu/ml/datasets/Adult
 
 
 
 
 In the dataset;
- There are missing values.
- There are 8 categorical and 6 numeric variables.
- Dependent variable is comprised of 2 classes(<=50K / >50K).


In this study;
- The missing values of the numeric variables are imputed by mean of each.
- The missing values of the categoric variables are imputed by mode(most frequent value) of each.
- The dependent variable is label encoded(<=50K == 0 / >50K == 1).
- The categoric independent variables are encoded by OneHotEncoder.
- The numeric independent variables are scaled by standardization method. 
- 5 different classification model which Random Forest, Logistic Regression, Decision Tree, SVM, and XGBOOST.
- Each model's accuracy criterion is evaluated by confusion matrix and accuracy score.
- Also to test each model's, certainity of accuracy score is examined k-Fold Cross Validation method.
