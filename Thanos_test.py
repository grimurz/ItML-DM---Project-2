# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 14:47:51 2020

@author: white
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from matplotlib.pyplot import figure, plot, subplot, title, xlabel, ylabel, show, clim
import seaborn as sns
from sklearn import model_selection
from toolbox_02450 import feature_selector_lr, bmplot


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

#Binarization of tobacco, alcohol in the classification data
tobacco_binary = binary_heart_data['tobacco']>0.5
binary_heart_data.insert(2, 'tobacco_binary', tobacco_binary, allow_duplicates = False)
alcohol_binary =binary_heart_data['alcohol']==0
binary_heart_data.insert(9, 'alcohol_binary', alcohol_binary, allow_duplicates = False)

binary_heart_data.tobacco_binary=binary_heart_data.tobacco_binary.astype(int)
binary_heart_data.alcohol_binary=binary_heart_data.alcohol_binary.astype(int)

#Binarization of tobacco, alcohol in the regression data
tobacco_binary = regression_heart_data['tobacco']>0.5
regression_heart_data.insert(2, 'tobacco_binary', tobacco_binary, allow_duplicates = False)
alcohol_binary =regression_heart_data['alcohol']==0
regression_heart_data.insert(9, 'alcohol_binary', alcohol_binary, allow_duplicates = False)

regression_heart_data.tobacco_binary=regression_heart_data.tobacco_binary.astype(int)
regression_heart_data.alcohol_binary=regression_heart_data.alcohol_binary.astype(int)

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
#sns.pairplot(regression_heart_data)
N, M = Xc.shape

# Clean up variables
del scaler_binary, scaler_reg, fam_history, fh, heart_data, unique_hist, historyDict






#---------------------------------------------
#MODEL TRAINING
'''
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size = 0.25, random_state = 0)

# Applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
Xc_train_LDA = lda.fit_transform(Xc_train, yc_train)
Xc_test_LDA = lda.transform(Xc_test)


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
log_classifier = LogisticRegression(C=1, n_jobs=-1, random_state = 0)
log_classifier.fit(Xc_train_LDA, yc_train)

'''
#-------------------------------------------------------------------------
## Crossvalidation
# Create crossvalidation partition for evaluation
K = 5
CV = model_selection.KFold(n_splits=K,shuffle=True)

# Initialize variables
Features = np.zeros((M,K))
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_fs = np.empty((K,1))
Error_test_fs = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))

k=0
for train_index, test_index in CV.split(Xc):
    
    # extract training and test set for current CV fold
    Xc_train = Xc[train_index,:]
    yc_train = yc[train_index]
    Xc_test = Xc[test_index,:]
    yc_test = yc[test_index]
    internal_cross_validation = 10
    
    # Compute squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(yc_train-yc_train.mean()).sum()/yc_train.shape[0]
    Error_test_nofeatures[k] = np.square(yc_test-yc_test.mean()).sum()/yc_test.shape[0]

    # Compute squared error with all features selected (no feature selection)
    m = lm.LinearRegression(fit_intercept=True).fit(X_train, y_train)
    Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

    # Compute squared error with feature subset selection
    textout = ''
    selected_features, features_record, loss_record = feature_selector_lr(X_train, y_train, internal_cross_validation,display=textout)
    
    Features[selected_features,k] = 1
    # .. alternatively you could use module sklearn.feature_selection
    if len(selected_features) is 0:
        print('No features were selected, i.e. the data (X) in the fold cannot describe the outcomes (y).' )
    else:
        m = lm.LinearRegression(fit_intercept=True).fit(X_train[:,selected_features], y_train)
        Error_train_fs[k] = np.square(y_train-m.predict(X_train[:,selected_features])).sum()/y_train.shape[0]
        Error_test_fs[k] = np.square(y_test-m.predict(X_test[:,selected_features])).sum()/y_test.shape[0]
    
        figure(k)
        subplot(1,2,1)
        plot(range(1,len(loss_record)), loss_record[1:])
        xlabel('Iteration')
        ylabel('Squared error (crossvalidation)')    
        
        subplot(1,3,3)
        bmplot(attributeNames, range(1,features_record.shape[1]), -features_record[:,1:])
        clim(-1.5,0)
        xlabel('Iteration')

    print('Cross validation fold {0}/{1}'.format(k+1,K))
    print('Train indices: {0}'.format(train_index))
    print('Test indices: {0}'.format(test_index))
    print('Features no: {0}\n'.format(selected_features.size))

    k+=1






# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
randc_forest = RandomForestClassifier(n_estimators = 500, criterion = 'entropy', max_depth=4, min_samples_leaf=4, min_samples_split=9, max_features="auto", bootstrap=False, random_state = 0, n_jobs= -1)
randc_forest.fit(Xc_train, yc_train)













'''

# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
xgb_classifier = XGBClassifier(learning_rate= 0.01, n_jobs=-1, random_state = 0)
xgb_classifier.fit(Xc_train_LDA, yc_train)

# Predicting the Test set results
log_pred = log_classifier.predict(Xc_test_LDA)


rfc_pred =randc_forest.predict(Xc_test)



# Predicting the Test set results
xgb_pred = xgb_classifier.predict(Xc_test_LDA)


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
log_accuracies = cross_val_score(estimator = log_classifier, X = Xc_train_LDA, y = yc_train, cv = 10, n_jobs = -1)


# Applying k-Fold Cross Validation

rfc_accuracies = cross_val_score(estimator = randc_forest, X = Xc_train, y = yc_train, cv = 10, n_jobs = -1)




# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
xgb_accuracies = cross_val_score(estimator = xgb_classifier, X = Xc_train_LDA, y = yc_train, cv = 10, n_jobs = -1)


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



# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = {'criterion': ['gini','entropy']}
            
               

grid_search = GridSearchCV(estimator = randc_forest,
                           param_grid = parameters,
                           scoring = 'neg_log_loss',
                           cv = 10,
                           n_jobs = -1)
grid_search_rfc = grid_search.fit(Xc_train, yc_train)
best_accuracy_rfc = grid_search.best_score_
best_parameters_rfc = grid_search.best_params_

'''

