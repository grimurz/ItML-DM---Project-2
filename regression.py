
from load_data import X#, y, binary_heart_data

import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn import model_selection

from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, subplot, hist, show


# Prepare data
y = np.array(X[:,3]).reshape(-1,1) # y = adiposity
X = X[:,[0,1,2, 4,5,6,7,8]] # exclude adiposity


### Regression, part a ###

#%% 1. Adiposity because it's well correlated? Show that we can use regression
#      to impute missing data?

# All attributes are used. Do we want some improved attribute selection?
# Standardization should strictly speaking happen within each fold. Not sure
# if it really matters in our case

# K-fold crossvalidation
K = 10
CV = model_selection.KFold(n_splits=K, shuffle=True, random_state=42)

# Init root-mean-square error (RMSE)
rmse = np.zeros(K)

k=0
for train_index, test_index in CV.split(X):

    # extract training and test set for current CV fold
    X_train, y_train = X[train_index,:], y[train_index]
    X_test, y_test = X[test_index,:], y[test_index]

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


#%% 2. Two-layer k-fold cross-validation

# Values of lambda
lambdas = np.logspace(-2, 4, 20)

# K-fold crossvalidation
K = 10
CV = model_selection.KFold(n_splits=K, shuffle=True, random_state=42)

# Init root-mean-square error (RMSE)
rmse_train = np.zeros([K,len(lambdas)])
rmse_test = np.zeros([K,len(lambdas)])

k=0
for train_index, test_index in CV.split(X):

    # extract training and test set for current CV fold
    X_train, y_train = X[train_index,:], y[train_index]
    X_test, y_test = X[test_index,:], y[test_index]

    # Fit for each lambda
    for i, lam in enumerate(lambdas):

        # Fit ridge regression model
        ridge_reg = Ridge(alpha=lam)
        ridge_reg.fit(X_train, y_train)

        # Compute model output:
        y_train_pred = ridge_reg.predict(X_train)
        y_test_pred = ridge_reg.predict(X_test)

        # Calculate error
        rmse_train[k,i] = np.sqrt(np.mean((y_train_pred-y_train)**2))
        rmse_test[k,i] = np.sqrt(np.mean((y_test_pred-y_test)**2))

    k+=1

# print('train RMSE:', np.round(np.mean(rmse_train,axis=0),4))
# print('test RMSE:', np.round(np.mean(rmse_test,axis=0),4))

train_error = np.mean(rmse_train,axis=0)
test_error = np.mean(rmse_test,axis=0)

min_error = np.min(test_error)
min_error_index = np.where(test_error == min_error)[0][0]

plt.figure(figsize=(8,8))
plt.semilogx(lambdas, train_error)
plt.semilogx(lambdas, test_error)
plt.semilogx(min_error_index, min_error, 'o')
plt.text(0.01, 0.6, "Minimum test error: " + str(np.round(min_error,4)) + ' at 1e' + str(np.round(np.log10(lambdas[min_error_index]),2)))
plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
plt.ylabel('Error rate (%)')
plt.title('Classification error')
plt.legend(['Training error','Test error','Test minimum'],loc='upper right')
plt.ylim([0.5, 1])
plt.grid()
plt.show()  





del k, K, ridge_reg, train_index, test_index
# del rmse


#%% Copy/pase bin, junk and other stuff

# X = X[np.argsort(X[:, 6])] # line plot gets otherwise messy

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


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