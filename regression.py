
from load_data import X#, y, binary_heart_data

import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, subplot, hist, show


# Prepare data
y = np.array(X[:,3]).reshape(-1,1) # y = adiposity
# X1 = X[:,[0,1,2, 4,5,6,7,8]] # exclude adiposity
# X1 = X[:,[1,2, 4,5,6,7,8]]  # 0: 0.537
# X1 = X[:,[2,4, 5,6,7,8]]  # 1: 0.5353
# X1 = X[:,[2, 5,6,7,8]]  # 4/5: 0.5346
X1 = X[:,[2, 6,7,8]]  # 5: 0.534   <---  best linreg according to primitive backwards selection
X2 = X[:,[2, 6,8]]  # 7: 0.5345   <---  best ridgereg (3rd order poly)
# X2 = X[:,[ 6,8]]  # 2: 0.5444
# X2 = X[:,[ 6 ]]  # 8: 0.6955


### REGRESSION, PART A ###

#%% 1. Adiposity because it's well correlated? Show that we can use regression
#      to impute missing data?

# All attributes are used. Do we want some improved attribute selection?
# Standardization should strictly speaking happen within each fold. Not sure
# if it really matters in our case. SWITCH FROM RMSE TO MSE?

# K-fold crossvalidation
K = 10
CV = model_selection.KFold(n_splits=K, shuffle=True, random_state=42)

# Init root-mean-square error (RMSE)
rmse = np.zeros(K)

k=0
for train_index, test_index in CV.split(X1):

    # extract training and test set for current CV fold
    X_train, y_train = X1[train_index,:], y[train_index]
    X_test, y_test = X1[test_index,:], y[test_index]

    # Fit ordinary least squares regression model
    lin_reg = lm.LinearRegression(fit_intercept=True)
    lin_reg.fit(X_train, y_train)
    
    # Compute model output:
    y_pred = lin_reg.predict(X_test)

    # Calculate error
    rmse[k] = np.sqrt(np.mean((y_pred-y_test)**2))

    k+=1

print('Linear regression RMSE:', np.round(np.mean(rmse),4))

del k, K, lin_reg, train_index, test_index
del rmse



#%% 2. 3rd order poly (Two-layer CV?)

# Values of lambda
lambdas = np.logspace(-3, 4, 50)

# K-fold crossvalidation
K = 10
CV = model_selection.KFold(n_splits=K, shuffle=True, random_state=42)

# Init root-mean-square error (RMSE)
rmse_train = np.zeros([K,len(lambdas)])
rmse_test = np.zeros([K,len(lambdas)])

k=0
# for train_index, test_index in CV.split(X2):
for train_index, test_index in CV.split(X2):

    # extract training and test set for current CV fold
    # X_train, y_train = X2[train_index,:], y2[train_index]
    # X_test, y_test = X2[test_index,:], y2[test_index]
    X_train, y_train = X2[train_index,:], y[train_index]
    X_test, y_test = X2[test_index,:], y[test_index]

    # Fit for each lambda
    for i, lam in enumerate(lambdas):

        # Fit ridge regression model
        ridge_reg = make_pipeline(PolynomialFeatures(3), Ridge(alpha=lam))
        ridge_reg.fit(X_train, y_train)

        # Compute model output:
        y_train_pred = ridge_reg.predict(X_train)
        y_test_pred = ridge_reg.predict(X_test)

        # Calculate error
        rmse_train[k,i] = np.sqrt(np.mean((y_train_pred-y_train)**2))
        rmse_test[k,i] = np.sqrt(np.mean((y_test_pred-y_test)**2))

    k+=1


train_error = np.mean(rmse_train,axis=0)
test_error = np.mean(rmse_test,axis=0)

min_error = np.min(test_error)
min_error_index = np.where(test_error == min_error)[0][0]

plt.figure(figsize=(8,8))
plt.semilogx(lambdas, train_error)
plt.semilogx(lambdas, test_error)
plt.semilogx(lambdas[min_error_index], min_error, 'o')
plt.text(0.007, 0.58, "Minimum test error: " + str(np.round(min_error,4)) + ' at 1e' + str(np.round(np.log10(lambdas[min_error_index]),2)))
plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
plt.ylabel('RMSE')
plt.title('Regression error')
plt.legend(['Training error','Test error','Test minimum'],loc='upper right')
plt.ylim([0.3, 1])
plt.grid()
plt.show()  

print('Ridge regression (inner) RMSE:', np.round(np.mean(min_error),4))


# # Fit ridge regression model, compute model output and calculate error:
# ridge_reg = make_pipeline(PolynomialFeatures(3), Ridge(alpha=lambdas[min_error_index]))
# ridge_reg.fit(X_train, y_train)
# y_te_pred = ridge_reg.predict(X_te)
# beep = np.sqrt(np.mean((y_te_pred-y_te)**2))
# print('Ridge regression (outer) RMSE:', np.round(np.mean(min_error),4))


del k, K, i, lam, ridge_reg, train_index, test_index
del rmse_train, rmse_test, min_error, min_error_index


#%% 3. 
'''
i    Get 2nd, 6th and 8th attribute
ii   Standardize data
iii  Use 3rd order polynomial interpolation on data
iv   Get prediction by running data through model
v    Unstandardize result from prediction
vi   profit???

https://scikit-learn.org/stable/auto_examples/linear_model/plot_polynomial_interpolation.html
'''


### REGRESSION, PART B ###

#%% 1.

K1 = 10
K2 = 10

#lambdas = ???
#hidden_units = ???

# Init errors
ann_error = np.zeros(K1) # articial neural network
rr_error = np.zeros(K1)  # ridge regression
bl_error = np.zeros(K1)  # baseline


CV1 = model_selection.KFold(n_splits=K1, shuffle=True, random_state=42)
CV2 = model_selection.KFold(n_splits=K2, shuffle=True, random_state=42)

# Outer CV for test data
k=0
for par_index, test_index in CV1.split(X):

    # extract training and test set for current CV fold
    X_par, y_par = X[par_index,:], y[par_index]
    X_test, y_test = X[test_index,:], y[test_index]

    # Inner loop for training and validation
    j=0
    for train_index, val_index in CV2.split(X_par):
    
        # extract training and test set for current CV fold
        X_train, y_train = X_par[train_index,:], y_par[train_index]
        X_val, y_val = X_par[val_index,:], y_par[val_index]

        # crunch that sweet training data

        # ANN
            
            
        # Ridge



        
        # validate that shiz and find those optimal hypers
        
        j+=1
    

    ### use test data here ###

    # ANN
        
        
    # Ridge


    # Baseline
    y_bl_pred = np.mean(y_test)
    bl_error[k] = np.mean((y_bl_pred-y_test)**2)  # mean square error

    k+=1
    
    
# Take mean from all three methods and compare?



del j, k, K1, K2, y_bl_pred
del par_index, train_index, val_index, test_index


#%% 2.









#%% 3.








#%% Copy/pase bin, junk and other stuff


# X2, X_te, y2, y_te = train_test_split(X, y, test_size = 0.2, random_state = 42)

# X = X[np.argsort(X[:, 6])] # line plot gets otherwise messy

# # polyreg, probably not used
# age = np.array(X[:,6]).reshape(-1,1)
# age_pow = age**2
# age2 = np.asarray(np.bmat('age, age_pow'))


### TEST 1 ### adi/age

# # Fit ordinary least squares regression model
# model = lm.LinearRegression(fit_intercept=True)
# model = model.fit(age,y)

# # Compute model output:
# y_est = model.predict(age)
# residual = y_est - y

# # Plot
# f = figure()
# plot(age,y,'.')
# plot(age,y_est,'-')

# figure()
# hist(residual,40)



### TEST 2 ### adi/(age,age^2)

# # Fit ordinary least squares regression model
# model = lm.LinearRegression(fit_intercept=True)
# model = model.fit(age2,y)

# # Compute model output:
# y_est = model.predict(age2)
# residual = y_est - y

# # Plot
# f = figure()
# plot(age,y,'.')
# plot(age,y_est,'-')

# figure()
# hist(residual,40)



# ### TEST 3 ### adi/X

# # Fit ordinary least squares regression model
# model = lm.LinearRegression(fit_intercept=True)
# model = model.fit(X,y)

# # Compute model output:
# y_est = model.predict(X)
# residual = y_est - y

# # Plot
# f = figure()
# plot(y,y_est,'.')

# figure()
# hist(residual,40)


# del model, f