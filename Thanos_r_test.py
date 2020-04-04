# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 15:03:00 2020

@author: white
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from matplotlib.pylab import figure, plot, xlabel, ylabel, legend, show,subplot, title, clim
from sklearn import model_selection, metrics
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
#regression_heart_data.drop('sbp', axis=1, inplace=True)
#regression_heart_data.drop('ldl', axis=1, inplace=True)
regression_heart_data.drop('age', axis=1, inplace=True)
#regression_heart_data.drop('famhist', axis=1, inplace=True)
#regression_heart_data.drop('typea', axis=1, inplace=True)
#regression_heart_data.drop('chd', axis=1, inplace=True)
#regression_heart_data.drop('obesity', axis=1, inplace=True)
#regression_heart_data.drop('tobacco', axis=1, inplace=True)
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
regression_attribute_names=regression_heart_data.columns
# Non-standardized data
Xns = binary_heart_data.to_numpy()

N, M = Xr.shape

# Clean up variables
del scaler_binary, scaler_reg, fam_history, fh, heart_data, unique_hist, historyDict




#---------------------------------------------
#MODEL TRAINING

'''

from sklearn.model_selection import train_test_split
Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size = 0.25, random_state = 0)

'''




## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
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
for train_index, test_index in CV.split(Xr):
    
    # extract training and test set for current CV fold
    Xr_train = Xr[train_index,:]
    yr_train = yr[train_index]
    Xr_test = Xr[test_index,:]
    yr_test = yr[test_index]
    internal_cross_validation = 10
    
    # Compute squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(yr_train-yr_train.mean()).sum()/yr_train.shape[0]
    Error_test_nofeatures[k] = np.square(yr_test-yr_test.mean()).sum()/yr_test.shape[0]

    # Compute squared error with all features selected (no feature selection)
    # Fitting Linear Regression to the dataset
    from sklearn.linear_model import LinearRegression
    lin_reg = LinearRegression(fit_intercept=True,n_jobs=-1)
    lin_reg.fit(Xr_train, yr_train)
    Error_train[k] = np.square(yr_train-lin_reg.predict(Xr_train)).sum()/yr_train.shape[0]
    Error_test[k] = np.square(yr_test-lin_reg.predict(Xr_test)).sum()/yr_test.shape[0]

    # Compute squared error with feature subset selection
    textout = ''
    selected_features, features_record, loss_record = feature_selector_lr(Xr_train, yr_train, internal_cross_validation,display=textout)
    
    Features[selected_features,k] = 1
    # .. alternatively you could use module sklearn.feature_selection
    if len(selected_features) is 0:
        print('No features were selected, i.e. the data (X) in the fold cannot describe the outcomes (y).' )
    else:
        lin_reg = LinearRegression(fit_intercept=True,n_jobs=-1)
        lin_reg.fit(Xr_train[:,selected_features], yr_train)
        Error_train_fs[k] = np.square(yr_train-lin_reg.predict(Xr_train[:,selected_features])).sum()/yr_train.shape[0]
        Error_test_fs[k] = np.square(yr_test-lin_reg.predict(Xr_test[:,selected_features])).sum()/yr_test.shape[0]
    
        figure(k)
        subplot(1,2,1)
        plot(range(1,len(loss_record)), loss_record[1:])
        xlabel('Iteration')
        ylabel('Squared error (crossvalidation)')    
        
        subplot(1,3,3)
        bmplot(regression_attribute_names, range(1,features_record.shape[1]), -features_record[:,1:])
        clim(-1.5,0)
        xlabel('Iteration')


    print('Cross validation fold {0}/{1}'.format(k+1,K))
    print('Train indices: {0}'.format(train_index))
    print('Test indices: {0}'.format(test_index))
    print('Features no: {0}\n'.format(selected_features.size))

    k+=1

# Display results
print('\n')
print('Linear regression without feature selection:\n')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('Linear regression with feature selection:\n')
print('- Training error: {0}'.format(Error_train_fs.mean()))
print('- Test error:     {0}'.format(Error_test_fs.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_fs.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test_fs.sum())/Error_test_nofeatures.sum()))

figure(k)
subplot(1,3,2)
bmplot(regression_attribute_names, range(1,Features.shape[1]+1), -Features)
clim(-1.5,0)
xlabel('Crossvalidation fold')
ylabel('Attribute')


# Inspect selected feature coefficients effect on the entire dataset and
# plot the fitted model residual error as function of each attribute to
# inspect for systematic structure in the residual

f=2 # cross-validation fold to inspect
ff=Features[:,f-1].nonzero()[0]
if len(ff) is 0:
    print('\nNo features were selected, i.e. the data (X) in the fold cannot describe the outcomes (y).' )
else:
    lin_reg = LinearRegression(fit_intercept=True,n_jobs=-1)
    lin_reg.fit(Xr_train[:,ff], yr_train)
    lin_reg_pred= lin_reg.predict(Xr[:,ff])
    residual=yr-lin_reg_pred
    
    figure(k+1, figsize=(12,6))
    title('Residual error vs. Attributes for features selected in cross-validation fold {0}'.format(f))
    for i in range(0,len(ff)):
       subplot(2,np.ceil(len(ff)/2.0),i+1)
       plot(Xr[:,ff[i]],residual,'.')
       xlabel(regression_attribute_names[ff[i]])
       ylabel('residual error')
    
    
show()



























# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
rand_forest = RandomForestRegressor(n_estimators = 290, max_depth=12, min_samples_split= 15,min_samples_leaf=2, max_features="auto",bootstrap=True, random_state = 0, n_jobs= -1)
rand_forest.fit(Xr_train, yr_train)

















































'''
# Predicting a new result with Linear Regression
lin_pred_train = lin_reg.predict(Xr_train)
lin_pred_test = lin_reg.predict(Xr_test)

# Predicting a new result with Random Forests
rf_pred = rand_forest.predict(Xr_test)




# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
lin_reg_accuracies = cross_val_score(estimator = lin_reg, X = Xr_train, y = yr_train, cv = 10, n_jobs = -1)
print("Accuracy: %0.2f (+/- %0.2f)" % (lin_reg_accuracies.mean(), lin_reg_accuracies.std() * 2))

# Applying k-Fold Cross Validation
rf_accuracies = cross_val_score(estimator = rand_forest, X = Xr_train, y = yr_train, cv = 10, n_jobs = -1)
print("Accuracy: %0.3f (+/- %0.3f)" % (rf_accuracies.mean(), rf_accuracies.std() * 2))




# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = { 'max_depth': [12,13],
               'n_estimators': [285,290,295]}
grid_search = GridSearchCV(estimator = rand_forest,
                           param_grid = parameters,
                           scoring = 'neg_root_mean_squared_error',
                           cv = 10,
                           n_jobs = -1)
grid_search_log = grid_search.fit(Xr_train, yr_train)
best_accuracy_log = grid_search.best_score_
best_parameters_log = grid_search.best_params_




#------------------------------------------------------------------------------------
#METRICS



#MAE is the easiest to understand, because it's the average error.
#MSE is more popular than MAE, because MSE "punishes" larger errors, which tends to be useful in the real world.
#RMSE is even more popular than MSE, because RMSE is interpretable in the "y" units.
#All of these are loss functions, because we want to minimize them.

print()
print()
print('MAE:', metrics.mean_absolute_error(yr_test, lin_pred_test))
print('MSE:', metrics.mean_squared_error(yr_test, lin_pred_test))
print('RMSE:', np.sqrt(metrics.mean_squared_error(yr_test, lin_pred_test)))

print('MAE:', metrics.mean_absolute_error(yr_test, rf_pred))
print('MSE:', metrics.mean_squared_error(yr_test, rf_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(yr_test, rf_pred)))

r2 = lin_reg.score(Xr_train,yr_train)
# Number of observations is the shape along axis 0
n = Xr_train.shape[0]
# Number of features (predictors, p) is the shape along axis 1
p = Xr_train.shape[1]

# We find the Adjusted R-squared using the formula
adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
adjusted_r2



#------------------------------------------------------------------------------------
#idle code

#Calc of decision trees error
tc = np.arange(2, 21, 1)
# Initialize variables
Error_train = np.empty((len(tc),1))
Error_test = np.empty((len(tc),1))

for i, t in enumerate(tc):
    # Fit decision tree classifier, Gini split criterion, different pruning levels
    rand_forest = RandomForestRegressor(n_estimators = 10, random_state = 0)
    rand_forest.fit(Xr_train, yr_train)

    # Evaluate classifier's misclassification rate over train/test data
    y_est_test = np.asarray(rand_forest.predict(Xr_test),dtype=int)
    y_est_train = np.asarray(rand_forest.predict(Xr_train), dtype=int)
    misclass_rate_test = sum(y_est_test != yr_test) / float(len(y_est_test))
    misclass_rate_train = sum(y_est_train != yr_train) / float(len(y_est_train))
    Error_test[i], Error_train[i] = misclass_rate_test, misclass_rate_train


f = figure()
plot(tc, Error_train*100)
plot(tc, Error_test*100)
xlabel('Model complexity (max tree depth)')
ylabel('Error (%)')
legend(['Error_train','Error_test'])
    
show()    
'''
#Alternative way to cals RMSE
'''
from sklearn.metrics import mean_squared_error
from math import sqrt

rms_lin = sqrt(mean_squared_error(yr_test, lin_pred))


rms_rf = sqrt(mean_squared_error(yr_test, rf_pred))

---------------------

# Predicting a new result with Linear Regression
lin_pred = lin_reg.predict(Xr_test[:,(1,3,7)])

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
lin_reg_accuracies = cross_val_score(estimator = lin_reg, X = X_Modeled, y = yr_train, cv = 10, n_jobs = -1)



# Building the optimal model using Backward Elimination
import statsmodels.api as sm
def backwardElimination(Xr_train, SL):
    numVars = len(Xr_train[0])
    temp = np.zeros((346,9))
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(yr_train, Xr_train).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = Xr_train[:, j]
                    Xr_train = np.delete(Xr_train, j, 1)
                    tmp_regressor = sm.OLS(yr_train, Xr_train).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((Xr_train, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
                    regressor_OLS.summary()
                    return Xr_train


SL = 0.05
X_opt = Xr_train[:, [0, 1, 2, 3, 4, 5,6,7,8]]
X_Modeled = backwardElimination(X_opt, SL)
'''








