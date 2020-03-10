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

# Importing the dataframe
dataframe = pd.read_csv('dataframe.csv')

# Impute missing values
for column in ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']:
    dataframe[column].fillna(dataframe[column].mode()[0], inplace=True)

for column in ['LoanAmount', 'ApplicantIncome', 'CoapplicantIncome']:
    dataframe[column].fillna(dataframe[column].mean(), inplace=True)
    
# =============================================================================
# dataframe['Gender'].fillna(dataframe['Gender'].mode()[0],inplace=True)
# =============================================================================

# =============================================================================
# dataframe.fillna(value = {'Gender': dataframe['Gender'].mode()[0],
#                         'Married': 'NO'
#                         },inplace = True)
# =============================================================================

# Convert all non-numeric values to number
cat=['Gender','Married','Dependents','Education','Self_Employed','Credit_History','Property_Area']
for var in cat:
    le = preprocessing.LabelEncoder()
    dataframe[var]=le.fit_transform(dataframe[var].astype('str'))
dataframe.dtypes

# Splitting the dataframe into the Training set and Test set
X = dataframe.iloc[:, 1:11].values
y = dataframe.iloc[:, 12].values
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

# Plot confusion matrix
from sklearn.metrics import plot_confusion_matrix
title='yes no'
class_names=['Y','N']
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(classifier, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)
plt.show()

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
