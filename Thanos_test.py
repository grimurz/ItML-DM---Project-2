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


'''
# Fitting  SVM to the Training set C=7 kerner='rbf', gamma=0.1 is the current optimal for cross_val_score
from sklearn.svm import SVC
SVM_classifier = SVC(C=7, kernel = 'linear',gamma=0.1, random_state = 0)
SVM_classifier.fit(Xc_train, yc_train)

 '''
 
#PROJECT 2, Classification, points 1-2-3 ----------------------

#LOGISTIC REGRESSION MODEL 

#Hyper-parameter training and selection
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# K-fold crossvalidation
K = 10



est=np.array([3])
for j in range(0,1):
    CV = model_selection.KFold(n_splits=K,shuffle=True, random_state = 42)

    # Initialize variables
    Error_train_base = np.empty((K,1))
    Error_test_base = np.empty((K,1))
    Error_train_GRID_log = np.empty((K,1))
    Error_test_GRID_log = np.empty((K,1))
    Error_train_RF = np.empty((K,1))
    Error_test_RF = np.empty((K,1))
    Error_train_SVM = np.empty((K,1))
    Error_test_SVM = np.empty((K,1))
    Error_train_gnb = np.empty((K,1))
    Error_test_gnb = np.empty((K,1))

    k=0
    # Outer loop
    for train_index, test_index in CV.split(Xc):
        
        # extract training and test set for current CV fold
        Xc_train_GRID_log = Xc[train_index,:]
        yc_train_GRID_log = yc[train_index]
        Xc_test_GRID_log = Xc[test_index,:]
        yc_test_GRID_log = yc[test_index]
        
    
        
        log_classifier_base = LogisticRegression(fit_intercept=False, n_jobs=-1)
        log_classifier_base.fit(Xc_train_GRID_log, yc_train_GRID_log)
        '''
        gnb = GaussianNB()
        gnb.fit(Xc_train_GRID_log, yc_train_GRID_log)
        
        gnb_test_pred=gnb.predict(Xc_test_GRID_log)
        gnb_train_pred=gnb.predict(Xc_train_GRID_log)
        
        
        misclass_rate_test_gnb = sum(gnb_test_pred != yc_test_GRID_log) / float(len(gnb_test_pred))
        misclass_rate_train_gnb = sum(gnb_train_pred != yc_train_GRID_log) / float(len(gnb_train_pred))
        Error_test_gnb[k], Error_train_gnb[k] = misclass_rate_test_gnb, misclass_rate_train_gnb
        '''
        # Fitting Random Forest  to the dataset
        
        rand_forest_simple = RandomForestClassifier(n_estimators = 400,  max_depth=4, min_samples_split=9,min_samples_leaf=2, max_features="auto",bootstrap=True, n_jobs= -1, random_state=13)
        rand_forest_simple.fit(Xc_train_GRID_log, yc_train_GRID_log)
        #Error calculation for the Random Forest model
        RF_test_pred = rand_forest_simple.predict(Xc_test_GRID_log)
        RF_train_pred = rand_forest_simple.predict(Xc_train_GRID_log)
        
    
        misclass_rate_test_RF = sum(RF_test_pred != yc_test_GRID_log) / float(len(RF_test_pred))
        misclass_rate_train_RF = sum(RF_train_pred != yc_train_GRID_log) / float(len(RF_train_pred))
        Error_test_RF[k], Error_train_RF[k] = misclass_rate_test_RF, misclass_rate_train_RF
        
        # Applying LDA
    
        lda = LDA(n_components = 2)
        Xc_train_LDA = lda.fit_transform(Xc_train_GRID_log, yc_train_GRID_log)
        Xc_test_LDA = lda.transform(Xc_test_GRID_log)
       
        log_classifier = LogisticRegression( C=1,  n_jobs=-1)
        log_classifier.fit(Xc_train_LDA, yc_train_GRID_log)
        
        
    
    
        #Grid search for Logistic Regression
    
        parameters = [{'C': [0.00000001,0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10]}]
        grid_search = GridSearchCV(estimator = log_classifier,
                                   param_grid = parameters,
                                   scoring = 'neg_log_loss',
                                   cv = 10,
                                   n_jobs = -1)
        grid_search= grid_search.fit(Xc_train_LDA, yc_train_GRID_log)
        
        #Error calculation for the baseline model
        misclass_rate_test_base = sum(log_classifier_base.predict(Xc_test_GRID_log) != yc_test_GRID_log) / float(len(log_classifier_base.predict(Xc_test_GRID_log)))
        misclass_rate_train_base = sum(log_classifier_base.predict(Xc_train_GRID_log) != yc_train_GRID_log) / float(len(log_classifier_base.predict(Xc_train_GRID_log)))
        Error_test_base[k], Error_train_base[k] = misclass_rate_test_base, misclass_rate_train_base
        #Error calculation for the LDA Logistic Regression model
        misclass_rate_test = sum(grid_search.predict(Xc_test_LDA) != yc_test_GRID_log) / float(len(grid_search.predict(Xc_test_LDA)))
        misclass_rate_train = sum(grid_search.predict(Xc_train_LDA) != yc_train_GRID_log) / float(len(grid_search.predict(Xc_train_LDA)))
        Error_test_GRID_log[k], Error_train_GRID_log[k] = misclass_rate_test, misclass_rate_train
        
        
        best_accuracy_log = grid_search.best_score_
        best_parameters_log = grid_search.best_params_
        print('Tuned Log ref CV fold {0}/{1}'.format(k+1,K))
        print('Train Error: {0}'.format(Error_train_GRID_log[k]))
        print('Test Error: {0}'.format(Error_test_GRID_log[k]))
        
        k+=1
        
        
        
        '''
        SVM_classifier = SVC()
        SVM_classifier.fit(Xc_train_LDA, yc_train_GRID_log)
        
        
        
        parameters = [{'C': [0.00000001,0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10], 'kernel': ['linear']},
              {'C': [0.00000001,0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
        grid_search = GridSearchCV(estimator = SVM_classifier,
                           param_grid = parameters,
                           scoring = 'neg_log_loss',
                           cv = 10,
                           n_jobs = -1)
        grid_search_SVM = grid_search.fit(Xc_train_LDA, yc_train_GRID_log)
        best_accuracy_SVM = grid_search.best_score_
        best_parameters_SVM = grid_search.best_params_

        misclass_rate_test_SVM = sum(grid_search_SVM.predict(Xc_test_LDA) != yc_test_GRID_log) / float(len(grid_search_SVM.predict(Xc_test_LDA)))
        misclass_rate_train_SVM = sum(grid_search_SVM.predict(Xc_train_LDA) != yc_train_GRID_log) / float(len(grid_search_SVM.predict(Xc_train_LDA)))
        Error_test_SVM[k], Error_train_SVM[k] = misclass_rate_test_SVM, misclass_rate_train_SVM
        
    '''
        
        
    
        
    print("For baseline Logistisc Regression: ")
    print('Train Error Accuracy({0}Kforld): {1}\n'.format(K,Error_train_base.T.mean(1)))
    print('Test Error Accuracy({0}Kforld): {1}\n'.format(K,Error_test_base.T.mean(1)))   
        
    print("For optimized Logistisc Regression: ")
    print('Train Error Accuracy({0}Kforld): {1}\n'.format(K,Error_train_GRID_log.T.mean(1)))
    print('Test Error Accuracy({0}Kforld): {1}\n'.format(K,Error_test_GRID_log.T.mean(1)))
    
    print("For Random Forests classification: ")
    print('Train Error Accuracy({0}Kforld): {1}\n'.format(K,Error_train_RF.T.mean()))
    print('Test Error Accuracy({0}Kforld): {1}\n'.format(K,Error_test_RF.T.mean()))
    #mean_train = Error_train_GRID_log.mean(1)
    #mean_test = Error_test_GRID_log.mean(1)
    C=[0.00000001,0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10]
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,6))
    plt.semilog(C, Error_train_base*100,label='Baseline train error')
    plt.semilog(C, Error_test_base*100,label='Baseline test error')
    plt.semilog(C, Error_train_GRID_log*100,label='Tuned logistic train error')
    plt.semilog(C, Error_test_GRID_log*100,label='Tuned logistic test error')
    plt.semilog(C, Error_train_RF.mean(1)*100,label='Simple Random forests train error')
    plt.semilog(C, Error_test_RF.mean(1)*100,label='Simple Random forests test error')
    xlabel('Test set')
    ylabel('Error (%), CV K={0}'.format(K))
    plt.legend(loc=0,shadow=True, fontsize='small')
    plt.title('Comparison of models random state {0}'.format(j))
    plt.grid() 
    plt.show()




Error_test_gnb



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

