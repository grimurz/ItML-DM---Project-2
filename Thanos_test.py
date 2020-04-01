# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 14:47:51 2020

@author: white
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns



#DATA PRE-PROCESSING


# Reading the dataset data and start of the pre-processing 
heart_data = pd.read_excel('Dataset.xlsx')
 
# We need to conduct an one-out-of-K transformation to the famhist column of
# data into a binary 0,1 form that is suitable to get fed into the PCA analysis
fam_history = heart_data['famhist']
unique_hist = np.unique(fam_history)

# We have created a dictionary where the attribute Present/Absent was
# codified into 0,1
historyDict = dict(zip(fam_history,[1,0]))

# The transformed array replaces the famhist values
fh = np.array([historyDict[cl] for cl in fam_history])
binary_heart_data = heart_data.copy()
binary_heart_data.famhist = fh
yc = heart_data[['chd']].to_numpy().squeeze()
yr= heart_data[['age']].to_numpy().squeeze()
binary_heart_data.drop('row.names', axis=1, inplace=True)
regression_heart_data = binary_heart_data.copy()
binary_heart_data.drop('chd', axis=1, inplace=True)
regression_heart_data.drop('age', axis=1, inplace=True)
# Data standardization: We scale our data so that each feature has
# a single unit of variance.
scaler_binary = StandardScaler()
scaler_binary.fit(binary_heart_data)
Xc = scaler_binary.transform(binary_heart_data)   # What about y?s

# Non-standardized data
Xns = binary_heart_data.to_numpy()


scaler_reg = StandardScaler()
scaler_reg.fit(regression_heart_data)
Xr = scaler_reg.transform(regression_heart_data)   # What about y?s

# Non-standardized data
Xns = binary_heart_data.to_numpy()
sns.pairplot(regression_heart_data)


# Clean up variables
del scaler_binary, scaler_reg, fam_history, fh, heart_data, unique_hist, historyDict






#---------------------------------------------
#MODEL TRAINING

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size = 0.25, random_state = 0)

# Applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
Xc_train = lda.fit_transform(Xc_train, yc_train)
Xc_test = lda.transform(Xc_test)


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
log_classifier = LogisticRegression(C=1, n_jobs=-1, random_state = 0)
log_classifier.fit(Xc_train, yc_train)

'''
# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
xgb_classifier = XGBClassifier(learning_rate= 0.01, n_jobs=-1, random_state = 0)
xgb_classifier.fit(Xc_train, yc_train)
'''
# Predicting the Test set results
log_pred = log_classifier.predict(Xc_test)
'''
# Predicting the Test set results
xgb_pred = xgb_classifier.predict(Xc_test)
'''

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
log_accuracies = cross_val_score(estimator = log_classifier, X = Xc_train, y = yc_train, cv = 10, n_jobs = -1)
'''
# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
xgb_accuracies = cross_val_score(estimator = xgb_classifier, X = Xc_train, y = yc_train, cv = 10, n_jobs = -1)

'''
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_log = confusion_matrix(yc_test, log_pred)



# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1,2,3,4,5,6,7,8,9]}]
grid_search = GridSearchCV(estimator = log_classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search_log = grid_search.fit(Xc_train, yc_train)
best_accuracy_log = grid_search.best_score_
best_parameters_log = grid_search.best_params_



