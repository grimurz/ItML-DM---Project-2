
from load_data import X_r, y_r

import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

# from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, subplot, hist, show

X0 = SelectKBest(f_regression, k=7).fit_transform(X_r, y_r)

# X1 = SelectKBest(mutual_info_regression, k=9).fit_transform(X_r, y_r)
X1 = SelectKBest(f_regression, k=7).fit_transform(X_r, y_r)
# X1 = X_r

X2 = SelectKBest(f_regression, k=7).fit_transform(X_r, y_r)
# X2 = X_r

### REGRESSION, PART A ###

#%% 1. 

# K-fold crossvalidation
K = 10
CV = model_selection.KFold(n_splits=K, shuffle=True, random_state=42)

# Init mean-square error (MSE)
mse = np.zeros(K)

k=0
for train_index, test_index in CV.split(X1):

    # extract training and test set for current CV fold
    X_train, y_train = X1[train_index,:], y_r[train_index]
    X_test, y_test = X1[test_index,:], y_r[test_index]

    # Fit ordinary least squares regression model
    lin_reg = lm.LinearRegression(fit_intercept=True)
    lin_reg.fit(X_train, y_train)
    
    # Compute model output:
    y_pred = lin_reg.predict(X_test)

    # Calculate error
    mse[k] = np.mean((y_pred-y_test)**2)

    k+=1

print('Linear regression MSE:', np.round(np.mean(mse),4))
print('Linear regression RMSE:', np.round(np.sqrt(np.mean(mse)),4),'\n')

del k, K, lin_reg, train_index, test_index
del mse



#%% 2. 3rd order poly 

# Values of lambda
lambdas = np.logspace(-3, 4, 50)

# K-fold crossvalidation
K = 10
CV = model_selection.KFold(n_splits=K, shuffle=True, random_state=42)



# Init mean-square error (MSE)
mse_train = np.zeros([K,len(lambdas)])
mse_test = np.zeros([K,len(lambdas)])

k=0
for train_index, test_index in CV.split(X2):

    # extract training and test set for current CV fold
    X_train, y_train = X2[train_index,:], y_r[train_index]
    X_test, y_test = X2[test_index,:], y_r[test_index]

    # Fit for each lambda
    for i, lam in enumerate(lambdas):

        # Fit ridge regression model
        ridge_reg = make_pipeline(PolynomialFeatures(3), Ridge(alpha=lam))
        ridge_reg.fit(X_train, y_train)

        # Compute model output:
        y_train_pred = ridge_reg.predict(X_train)
        y_test_pred = ridge_reg.predict(X_test)

        # Calculate error
        mse_train[k,i] = np.mean((y_train_pred-y_train)**2)
        mse_test[k,i] = np.mean((y_test_pred-y_test)**2)

    k+=1


train_error = np.mean(mse_train,axis=0)
test_error = np.mean(mse_test,axis=0)

min_error = np.min(test_error)
min_error_index = np.where(test_error == min_error)[0][0]

plt.figure(figsize=(8,8))
plt.semilogx(lambdas, np.sqrt(train_error))
plt.semilogx(lambdas, np.sqrt(test_error))
# plt.semilogx(lambdas, train_error)
# plt.semilogx(lambdas, test_error)
plt.semilogx(lambdas[min_error_index], np.sqrt(min_error), 'o')
plt.text(1, 10, "Minimum test error: " + str(np.round(np.sqrt(min_error),2)) + ' at 1e' + str(np.round(np.log10(lambdas[min_error_index]),2)))
plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
plt.ylabel('RMSE')
plt.title('Regression error')
plt.legend(['Training error','Test error','Test minimum'],loc='upper right')
plt.ylim([5, 15])
# plt.ylim([60, 120])
plt.grid()
plt.show()  

print('Ridge regression MSE:', np.round(np.mean(min_error),4))
print('Ridge regression RMSE:', np.round(np.sqrt(np.mean(min_error)),4))
print('lambda:', np.round(lambdas[min_error_index],4))



del k, K, i, lam, ridge_reg, train_index, test_index
# del mse_train, mse_test, min_error, min_error_index


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

#%% 1. K1/K2 and CV1/CV2 redundant? 

K1 =  2 # 10
K2 =  4 # 10

# Init hyperparameters
hidden_units = np.arange(start = 1, stop = 20, step = 3)
lambdas = np.logspace(-3, 4, 50)

# Init errors
ann_error = np.zeros(K1) # articial neural network
rr_error = np.zeros(K1)  # ridge regression
bl_error = np.zeros(K1)  # baseline


CV1 = model_selection.KFold(n_splits=K1, shuffle=True, random_state=42)
CV2 = model_selection.KFold(n_splits=K2, shuffle=True, random_state=43) # reduntant?

# Outer CV for test data
k=0
for par_index, test_index in CV1.split(X0):

    # extract training and test set for current CV fold
    X_par, y_par = X0[par_index,:], y_r[par_index]
    X_test, y_test = X0[test_index,:], y_r[test_index]

    # Init RMSE
    ann_error_val = np.zeros([K2,len(hidden_units)])
    rr_error_val = np.zeros([K2,len(lambdas)])
    
    # Init optimal lambda & optimal h
    h_opt = np.zeros(K2)
    lambda_opt = np.zeros(K2)
    
    # Init min error
    min_error_ann_val = np.zeros(K2)
    min_error_rr_val = np.zeros(K2)

    # Inner loop for training and validation
    j=0
    for train_index, val_index in CV2.split(X_par):
        
        # extract training and test set for current CV fold
        X_train, y_train = X_par[train_index,:], y_par[train_index]
        X_val, y_val = X_par[val_index,:], y_par[val_index]

        ##### crunch that sweet training data #####

        # ANN training
  

          
            

        # Ridge training
        for i, lam in enumerate(lambdas):
    
            # Fit ridge regression model
            ridge_reg = make_pipeline(PolynomialFeatures(3), Ridge(alpha=lam))
            ridge_reg.fit(X_train, y_train)
    
            # Compute model output:
            y_val_pred = ridge_reg.predict(X_val)
    
            # Calculate error (RMSE)
            rr_error_val[j,i] = np.sqrt(np.mean((y_val_pred-y_val)**2))

        mean_rr_error_val = np.mean(rr_error_val,axis=0)
        min_error_rr_val[j] = np.min(mean_rr_error_val)

        min_error_rr_index = np.where(mean_rr_error_val == min_error_rr_val[j])[0][0]
        lambda_opt[j] = lambdas[min_error_rr_index]
    
        print('\nK1:',k+1,' K2:',j+1)
        print('min rr RMSE error:', np.round(min_error_rr_val[j],4))
        print('opt lambda:', np.round(lambdas[min_error_rr_index],4))
    
    
        # Temp visualization, to be commented
        plt.figure(figsize=(8,8))
        plt.semilogx(lambdas, mean_rr_error_val)
        plt.semilogx(lambdas[min_error_rr_index], min_error_rr_val[j], 'o')
        plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
        plt.ylabel('RMSE')
        plt.title('Regression error')
        plt.legend(['Val error','Val minimum'],loc='upper right')
        plt.ylim([0, 30])
        plt.grid()
        plt.show()  

        
        j+=1

    print('\nmean rr val error', np.round(np.mean(min_error_rr_val),4))
    print('mean lambda', np.round(np.mean(lambda_opt),4))


    ##### test data here #####

    # ANN testing

    # ann_error[k] =
        
    
    # Ridge testing
    ridge_reg.fit(X_par, y_par)
    y_test_pred = ridge_reg.predict(X_test)
    rr_error[k] = np.sqrt(np.mean((y_test_pred-y_test)**2))

    # Baseline testing
    y_bl_pred = np.mean(y_test)
    bl_error[k] = np.sqrt(np.mean((y_bl_pred-y_test)**2))  # root mean square error

    k+=1
    
    
# Take mean from all three methods and compare?
print('\n')
print('final ann error:', np.round(ann_error,2))
print('final rr error:', np.round(rr_error,2))
print('final bl error:', np.round(bl_error,2))

print('\nfinal means:', np.round(np.mean(ann_error),2), np.round(np.mean(rr_error),2), np.round(np.mean(bl_error),2))


del i, j, k, K1, K2, y_bl_pred, lam
del par_index, train_index, val_index, test_index
# del hidden_units, lambdas


#%% 2.









#%% 3.








#%% Copy/paste bin, junk and other stuff



# Prepare data
# y = np.array(X[:,3]).reshape(-1,1) # y = adiposity
# X0 = X[:,[0,1,2, 4,5,6,7,8]] # exclude adiposity
# # X1 = X[:,[1,2, 4,5,6,7,8]]  # 0: 0.537
# # X1 = X[:,[2,4, 5,6,7,8]]  # 1: 0.5353
# # X1 = X[:,[2, 5,6,7,8]]  # 4/5: 0.5346
# X1 = X[:,[2, 6,7,8]]  # 5: 0.534   <---  best linreg according to primitive backwards selection
# X2 = X[:,[2, 6,8]]  # 7: 0.5345   <---  best ridgereg (3rd order poly)
# # X2 = X[:,[ 6,8]]  # 2: 0.5444
# # X2 = X[:,[ 6 ]]  # 8: 0.6955



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