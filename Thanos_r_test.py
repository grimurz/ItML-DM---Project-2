# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 15:03:00 2020

@author: white
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import metrics



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

# Non-standardized data
Xns = binary_heart_data.to_numpy()



# Clean up variables
del scaler_binary, scaler_reg, fam_history, fh, heart_data, unique_hist, historyDict




#---------------------------------------------
#MODEL TRAINING



from sklearn.model_selection import train_test_split
Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size = 0.25, random_state = 0)





'''
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



# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression(n_jobs=-1)
lin_reg.fit(Xr_train, yr_train)


# Predicting a new result with Linear Regression
lin_pred = lin_reg.predict(Xr_test)


'''
# Predicting a new result with Linear Regression
lin_pred = lin_reg.predict(Xr_test[:,(1,3,7)])

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
lin_reg_accuracies = cross_val_score(estimator = lin_reg, X = X_Modeled, y = yr_train, cv = 10, n_jobs = -1)
'''

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
lin_reg_accuracies = cross_val_score(estimator = lin_reg, X = Xr_train, y = yr_train, cv = 10, n_jobs = -1)


#------------------------------------------------------------------------------------
#METRICS


#MAE is the easiest to understand, because it's the average error.
#MSE is more popular than MAE, because MSE "punishes" larger errors, which tends to be useful in the real world.
#RMSE is even more popular than MSE, because RMSE is interpretable in the "y" units.
#All of these are loss functions, because we want to minimize them.

print()
print()
print('MAE:', metrics.mean_absolute_error(yr_test, lin_pred))
print('MSE:', metrics.mean_squared_error(yr_test, lin_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(yr_test, lin_pred)))

r2 = lin_reg.score(Xr_train,yr_train)
# Number of observations is the shape along axis 0
n = Xr_train.shape[0]
# Number of features (predictors, p) is the shape along axis 1
p = Xr_train.shape[1]

# We find the Adjusted R-squared using the formula
adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
adjusted_r2