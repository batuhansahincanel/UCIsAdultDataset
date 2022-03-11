#   IMPORTING THE LIBRARIES
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score

#Importing the dataset
dataset = pd.read_csv("adult.data", sep=",", names=["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                                                    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
                                                    "hours-per-week", "native-country", "income"])

#Defining X and y variables
y = dataset.iloc[:,-1].values
X = dataset.iloc[:,:-1]


#Spreading X variables as numeric and categoric data
X_num = X.select_dtypes(include=np.number)
X_cat = X.select_dtypes(exclude=np.number)


#   PREPROCESSING
#Handling with missing values
imputer_num = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer_cat = SimpleImputer(missing_values=np.nan, strategy="most_frequent")

X_num = imputer_num.fit_transform(X_num)
X_cat = imputer_cat.fit_transform(X_cat)

#Encoding the classes of the y variable as <=50 == 0 / >50 == 1
le_y = LabelEncoder()
y = le_y.fit_transform(y)

#Encoding the categorical data of the X variables 
ohe = OneHotEncoder()
X_cat = ohe.fit_transform(X_cat).toarray()

#Scaling the numeric data of the X variables with standardization method
sc = StandardScaler()
X_num = sc.fit_transform(X_num)

#Gathering together all X variables
X = np.concatenate((X_num, X_cat), axis=1)

#Spreading the data as train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)



#  TRAINING DIFFERENT CLASSIFICATION MODELS, PREDICTING AND EVALUATING BY ACCURACY 
#   ***RandomForest***
classifier_rf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
classifier_rf.fit(X_train, y_train)

y_pred_rf = classifier_rf.predict(X_test)

#Accuracy Score and Confusion Matrix of the Random Forest Model
ac_rf = accuracy_score(y_test, y_pred_rf)
cm_rf = confusion_matrix(y_test, y_pred_rf)

#Examining the accuracy of the Random Forest Model with k-Fold Cross Validation 
accuracies_rf = cross_val_score(classifier_rf, X, y, cv = 10, n_jobs = -1)
accuracies_rf_mean = accuracies_rf.mean()*100
accuracies_rf_std = accuracies_rf.std()*100



#   ***LogisticRegression***
classifier_lr = LogisticRegression(n_jobs=-1, random_state=0)
classifier_lr.fit(X_train, y_train)

y_pred_lr = classifier_lr.predict(X_test)

#Accuracy Score and Confusion Matrix of the Logistic Regression Model
ac_lr = accuracy_score(y_test, y_pred_lr)
cm_lr = confusion_matrix(y_test, y_pred_lr)

#Examining the accuracy of the Logistic Regression Model with k-Fold Cross Validation 
accuracies_lr = cross_val_score(classifier_lr, X, y, cv=10, n_jobs=-1)
accuracies_lr_mean = accuracies_lr.mean()*100
accuracies_lr_std = accuracies_lr.std()*100



# ***DecisionTree***
classifier_dt = DecisionTreeClassifier(criterion = 'entropy', random_state=0)
classifier_dt.fit(X_train, y_train)

y_pred_dt = classifier_dt.predict(X_test)

#Accuracy Score and Confusion Matrix of the Decision Tree Model
ac_dt = accuracy_score(y_test, y_pred_dt)
cm_dt = confusion_matrix(y_test, y_pred_dt)

#Examining the accuracy of the Decision Tree Model with k-Fold Cross Validation 
accuracies_dt = cross_val_score(classifier_dt, X, y, cv=10, n_jobs=-1)
accuracies_dt_mean = accuracies_dt.mean()*100
accuracies_dt_std = accuracies_dt.std()*100



# ***SVM***
classifier_svm = SVC(kernel = 'rbf', random_state=0)
classifier_svm.fit(X_train, y_train)

y_pred_svm = classifier_svm.predict(X_test)

#Accuracy Score and Confusion Matrix of the SVM Model
ac_svm = accuracy_score(y_test, y_pred_svm)
cm_svm = confusion_matrix(y_test, y_pred_svm)

#Examining the accuracy of the SVM Model with k-Fold Cross Validation 
accuracies_svm = cross_val_score(classifier_svm, X, y, cv=10, n_jobs=-1)
accuracies_svm_mean = accuracies_svm.mean()*100
accuracies_svm_std = accuracies_svm.std()*100



# ***XGBOOST***
classifier_xgb = XGBClassifier()
classifier_xgb.fit(X_train, y_train)

y_pred_xgb = classifier_xgb.predict(X_test)

#Accuracy Score and Confusion Matrix of the XGBOOST Model
ac_xgb = accuracy_score(y_test, y_pred_xgb)
cm_xgb = confusion_matrix(y_test, y_pred_xgb)

#Examining the accuracy of the XGBOOST Model with k-Fold Cross Validation 
accuracies_xgb = cross_val_score(classifier_xgb, X, y, cv=10, n_jobs=-1)
accuracies_xgb_mean = accuracies_xgb.mean()*100
accuracies_xgb_std = accuracies_xgb.std()*100









