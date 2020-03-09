# Logistic Regression Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, jaccard_score, log_loss, f1_score

# Importing the dataset
dataset = pd.read_csv('dataset.csv')

# Impute missing values
for column in ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']:
    dataset[column].fillna(dataset[column].mode()[0], inplace=True)

for column in ['LoanAmount', 'ApplicantIncome', 'CoapplicantIncome']:
    dataset[column].fillna(dataset[column].mean(), inplace=True)
    
# =============================================================================
# dataset['Gender'].fillna(dataset['Gender'].mode()[0],inplace=True)
# =============================================================================

# =============================================================================
# dataset.fillna(value = {'Gender': dataset['Gender'].mode()[0],
#                         'Married': 'NO'
#                         },inplace = True)
# =============================================================================

# Convert all non-numeric values to number
cat=['Gender','Married','Dependents','Education','Self_Employed','Credit_History','Property_Area']
for var in cat:
    le = preprocessing.LabelEncoder()
    dataset[var]=le.fit_transform(dataset[var].astype('str'))
dataset.dtypes

# Splitting the dataset into the Training set and Test set
X = dataset.iloc[:, 1:11].values
y = dataset.iloc[:, 12].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Results
le = preprocessing.LabelEncoder()
y_test=le.fit_transform(y_test.astype('str'))
y_pred=le.fit_transform(y_pred.astype('str'))

#classifier.score(X_test, y_test)
ac=accuracy_score(y_test, y_pred)
print('accuracy_score', ac)

f1=f1_score(y_test, y_pred, average='binary')
print('f1_score', f1)

js=jaccard_score(y_test, y_pred, average='binary')
print('jaccard_score', js)

y_pred2 = classifier.predict_proba(X_test)
#print(y_pred2)

ll = log_loss(y_test, y_pred2)
print('log_loss', ll)






# Refferences
# =============================================================================
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html#sklearn.metrics.jaccard_score
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html#sklearn.metrics.log_loss
# =============================================================================
