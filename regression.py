
from load_data import X_r, y_r

import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.linear_model import Ridge, Lasso
from sklearn import model_selection
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_regression

import torch
from toolbox_02450 import *


# X0 = SelectKBest(f_regression, k=7).fit_transform(X_r, y_r)

# # X1 = SelectKBest(mutual_info_regression, k=9).fit_transform(X_r, y_r)
# X1 = SelectKBest(f_regression, k=7).fit_transform(X_r, y_r)
# # X1 = X_r

# X2 = SelectKBest(f_regression, k=7).fit_transform(X_r, y_r)
# # X2 = X_r

### REGRESSION, PART A ###

#%% 1. 

# K-fold crossvalidation
K = 10
CV = model_selection.KFold(n_splits=K, shuffle=True, random_state=42)

# Init mean-square error (MSE)
mse = np.zeros(K)

k=0
for train_index, test_index in CV.split(X_r):

    # extract training and test set for current CV fold
    X_train, y_train = X_r[train_index,:], y_r[train_index]
    X_test, y_test = X_r[test_index,:], y_r[test_index]

    # Fit ordinary least squares regression model
    lin_reg = lm.LinearRegression(fit_intercept=True)
    lin_reg.fit(X_train, y_train)
    
    # Compute model output:
    y_pred = lin_reg.predict(X_test)

    # Calculate error
    mse[k] = np.mean((y_pred-y_test)**2)

    k+=1

# print('Linear regression MSE:', np.round(np.mean(mse),4))
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
for train_index, test_index in CV.split(X_r):

    # extract training and test set for current CV fold
    X_train, y_train = X_r[train_index,:], y_r[train_index]
    X_test, y_test = X_r[test_index,:], y_r[test_index]

    # Fit for each lambda
    for i, lam in enumerate(lambdas):

        # Fit ridge regression model
        ridge_reg = make_pipeline(PolynomialFeatures(2), Ridge(alpha=lam))
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
plt.semilogx(lambdas[min_error_index], np.sqrt(min_error), 'o')
plt.text(1, 10, "Minimum test error: " + str(np.round(np.sqrt(min_error),2)) + ' at 1e' + str(np.round(np.log10(lambdas[min_error_index]),2)))
plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
plt.ylabel('RMSE')
plt.title('Regression error')
plt.legend(['Training error','Test error','Test minimum'],loc='upper right')
plt.ylim([5, 15])
plt.grid()
plt.show()  

# print('Ridge regression MSE:', np.round(np.mean(min_error),4))
print('Ridge regression RMSE:', np.round(np.sqrt(np.mean(min_error)),4))
print('lambda:', np.round(lambdas[min_error_index],4))



del k, K, i, lam, ridge_reg, train_index, test_index
del mse_train, mse_test, min_error, min_error_index


#%% 3. 
'''
i    Select best attributes?
ii   Standardize data
iii  Use 3rd order polynomial interpolation on data
iv   Get prediction by running data through model
v    Unstandardize result from prediction(?)
vi   profit???

https://scikit-learn.org/stable/auto_examples/linear_model/plot_polynomial_interpolation.html
'''


### REGRESSION, PART B ###

#%% 1. K1/K2 and CV1/CV2 redundant? 
from sklearn.linear_model import Lasso
K1 =  10
K2 =  10

# Init hyperparameters
hidden_units = np.arange(start = 1, stop = 10, step = 2)
lambdas = np.logspace(-3, 4, 50)

# Init optimal hyperparameters
h_opt = np.zeros(K1).astype(int)
lambda_opt = np.zeros(K1)

# Init errors
nn_error = np.zeros(K1)  # articial neural network
rr_error = np.zeros(K1)  # ridge regression
bl_error = np.zeros(K1)  # baseline
nn_error_val_tot = np.zeros(K1)
rr_error_val_tot = np.zeros(K1)

# Init statistic evaluation
nn_rr = []
nn_bl = []
rr_bl = []
loss = 2

# Cross-validation
CV1 = model_selection.KFold(n_splits=K1, shuffle=True, random_state=42)
CV2 = model_selection.KFold(n_splits=K2, shuffle=True, random_state=43) # redundant?

# ANN model
N, M = X_r.shape

# Parameters for neural network 
n_replicates = 1       # number of networks trained in each k-fold
max_iter = 5000



##### Outer CV for test data #####
k=0
for par_index, test_index in CV1.split(X_r):

    # extract training and test set for current CV fold
    X_par, y_par = X_r[par_index,:], np.expand_dims(y_r, axis=1)[par_index]
    X_test, y_test = X_r[test_index,:], np.expand_dims(y_r, axis=1)[test_index]

    # Init RMSE
    nn_error_val = np.zeros([K2,len(hidden_units)])
    rr_error_val = np.zeros([K2,len(lambdas)])
    
    # Init optimal lambda & optimal h
    h_opt_val = np.zeros(K2)
    lambda_opt_val = np.zeros(K2)
    
    # Init min error
    min_error_nn_val = np.zeros(K2)
    min_error_rr_val = np.zeros(K2)



    # temp test graph
    plt.figure(figsize=(8,8))



    ##### Inner loop for training and validation #####
    j=0
    for train_index, val_index in CV2.split(X_par):
        
        # extract training and test set for current CV fold
        X_train, y_train = X_par[train_index,:], y_par[train_index]
        X_val, y_val = X_par[val_index,:], y_par[val_index]

        # Convert to tensors
        X_nn_train = torch.Tensor(X_train)
        y_nn_train = torch.Tensor(y_train)
        X_nn_val = torch.Tensor(X_val)


        ##### ANN training #####
        for i, h in enumerate(hidden_units):

            # Define the model
            model = lambda: torch.nn.Sequential(
                                torch.nn.Linear(M, h), #M features to n_hidden_units
                                torch.nn.Tanh(), # 1st transfer function,

                                torch.nn.Linear(h, h),   # torch.nn.ReLU()   torch.nn.Tanh()
                                torch.nn.ReLU(),
            
                                torch.nn.Linear(h, 1), # n_hidden_units to 1 output neuron
                                # no final tranfer function, i.e. "linear output"
                                )
            loss_fn = torch.nn.MSELoss()
            
            # Train the net on training data
            net, final_loss, learning_curve = train_neural_net(model,
                                                                loss_fn,
                                                                X=X_nn_train,
                                                                y=y_nn_train,
                                                                n_replicates=n_replicates,
                                                                max_iter=max_iter)
            
            # Determine estimated class labels for test set
            y_nn_val_pred = net(X_nn_val).detach().numpy()

            # Calculate error (RMSE)
            nn_error_val[j,i] = np.sqrt(np.mean((y_nn_val_pred-y_val)**2))

        # mean_nn_error_val = np.mean(nn_error_val,axis=0) # WRONG
        min_error_nn_val[j] = np.min(nn_error_val[j])
        min_error_nn_index = np.where(nn_error_val[j] == min_error_nn_val[j])[0][0]
        h_opt_val[j] = hidden_units[min_error_nn_index]
        
        

        ##### Ridge training #####
        for i, lam in enumerate(lambdas):
            
            # lasso_reg = make_pipeline(PolynomialFeatures(2), Lasso(alpha=lam))
            # lasso_reg.fit(X_train, y_train)
    
            # Fit ridge regression model
            ridge_reg = make_pipeline(PolynomialFeatures(2), Ridge(alpha=lam))
            ridge_reg.fit(X_train, y_train)
    
            # Compute model output:
            # y_val_pred = lasso_reg.predict(X_val)
            y_val_pred = ridge_reg.predict(X_val)
    
            # Calculate error (RMSE)
            rr_error_val[j,i] = np.sqrt(np.mean((y_val_pred-y_val)**2))

        # mean_rr_error_val = np.mean(rr_error_val,axis=0) # WRONG
        min_error_rr_val[j] = np.min(rr_error_val[j]) # np.min(mean_rr_error_val)
        min_error_rr_index = np.where(rr_error_val[j] == min_error_rr_val[j])[0][0]
        lambda_opt_val[j] = lambdas[min_error_rr_index]
    
    

        print('\nK1:',k+1,' K2:',j+1)
        print('min rr RMSE error:', np.round(min_error_rr_val[j],4))
        print('min nn RMSE error:', np.round(min_error_nn_val[j],4))
        print('opt lambda:', np.round(lambdas[min_error_rr_index],4)) # <--- ATTN!
        print('opt h:', np.round(hidden_units[min_error_nn_index],4)) # <--- ATTN!


    
    
        # # # Temp visualization, to be commented
        # plt.figure(figsize=(8,8))
        # plt.plot(hidden_units, mean_nn_error_val)
        # plt.plot(hidden_units[min_error_nn_index], min_error_nn_val[j], 'o')
        # plt.xlabel('Hidden units')
        # plt.ylabel('RMSE')
        # plt.title('ANN - error')
        # plt.legend(['Val error','Val minimum'],loc='upper right')
        # plt.ylim([0, 30])
        # plt.grid()
        # plt.show()  
        
        # plt.figure(figsize=(8,8))
        # plt.semilogx(lambdas, mean_rr_error_val)
        # plt.semilogx(lambdas[min_error_rr_index], min_error_rr_val[j], 'o')
        # plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
        # plt.ylabel('RMSE')
        # plt.title('Ridge regression - error')
        # plt.legend(['Val error','Val minimum'],loc='upper right')
        # plt.ylim([0, 30])
        # plt.grid()
        # plt.show()  

        plt.plot(hidden_units, nn_error_val[j])
        plt.plot(hidden_units[min_error_nn_index], min_error_nn_val[j], 'o')
        
        
        j+=1


    # temp test graph
    plt.xlabel('Hidden units')
    plt.ylabel('RMSE')
    plt.title('ANN - error - Outer CV fold no.'+str(k+1))
    plt.legend(['Val error','Val minimum'],loc='upper right')
    plt.ylim([0, 30])
    plt.grid()
    plt.show()  



    h_opt[k] = np.round(np.mean(h_opt_val)).astype(int)
    lambda_opt[k] = np.mean(lambda_opt_val)

    # Collect mean of min errors
    nn_error_val_tot[k] = np.mean(min_error_nn_val)
    rr_error_val_tot[k] = np.mean(min_error_rr_val)


    print('\nmean rr val error', np.round(np.mean(min_error_rr_val),4))
    print('mean nn val error', np.round(np.mean(min_error_nn_val),4))
    print('mean lambda', np.round(lambda_opt[k],4))
    print('mean h', h_opt[k])
    # print('most frequent h', np.argmax(np.bincount(h_opt_val.astype(int))))



    ##### Validation using test data #####

    # ANN testing    
    X_nn_par = torch.Tensor(X_par)
    y_nn_par = torch.Tensor(y_par)
    X_nn_test = torch.Tensor(X_test)

    model = lambda: torch.nn.Sequential(
                        torch.nn.Linear(M, h_opt[k]),
                        torch.nn.Tanh(),
                        torch.nn.Linear(h_opt[k], h_opt[k]),
                        torch.nn.ReLU(),
                        torch.nn.Linear(h_opt[k], 1),
                        )
    loss_fn = torch.nn.MSELoss()

    net, final_loss, learning_curve = train_neural_net(model,
                                                        loss_fn,
                                                        X=X_nn_par,
                                                        y=y_nn_par,
                                                        n_replicates=n_replicates,
                                                        max_iter=max_iter)
    y_nn_test_pred = net(X_nn_test).detach().numpy()
    nn_error[k] = np.sqrt(np.mean((y_nn_test_pred-y_test)**2))
        
    
    # Ridge testing
    ridge_reg = make_pipeline(PolynomialFeatures(2), Ridge(alpha=lambda_opt[k]))
    ridge_reg.fit(X_par, y_par)
    y_test_pred = ridge_reg.predict(X_test)
    rr_error[k] = np.sqrt(np.mean((y_test_pred-y_test)**2))


    # Baseline testing
    lin_reg = lm.LinearRegression(fit_intercept=True)
    lin_reg.fit(X_par, y_par)
    y_bl_pred = lin_reg.predict(X_test)
    bl_error[k] = np.sqrt(np.mean((y_bl_pred-y_test)**2))  # root mean square error



    ##### Statistic evaluation #####
    nn_rr.append( np.mean( np.abs( y_nn_test_pred-y_test ) ** loss - np.abs( y_test_pred-y_test) ** loss ) )
    nn_bl.append( np.mean( np.abs( y_nn_test_pred-y_test ) ** loss - np.abs( y_bl_pred-y_test) ** loss ) )
    rr_bl.append( np.mean( np.abs( y_test_pred-y_test ) ** loss - np.abs( y_bl_pred-y_test) ** loss ) )



    k+=1
    
print('\nFinal validation error:')
print('nn val:', np.round(np.mean(nn_error_val_tot),4))
print('rr val:', np.round(np.mean(rr_error_val_tot),4))
    
# Take mean from all three methods and compare (NO PEEKING)
print('\n')
print('estimated nn error:', np.round(nn_error,2))
print('estimated rr error:', np.round(rr_error,2))
print('estimated bl error:', np.round(bl_error,2))
print('optimal hidden units:', h_opt)
print('optimal lambdas:', np.round(lambda_opt,2))

print('\nfinal means:\nNN:', np.round(np.mean(nn_error),2), '\nRR:', np.round(np.mean(rr_error),2), '\nBL:', np.round(np.mean(bl_error),2))



#%% 2.

# Pie, see above


#%% 3.

alpha = 0.05
rho = 1/K1

# p-values for the null hypothesis that the two models have the same performance

# NN vs RR
nn_rr_p, nn_rr_CI = correlated_ttest(nn_rr, rho, alpha=alpha)

# NN vs BL
nn_bl_p, nn_bl_CI = correlated_ttest(nn_bl, rho, alpha=alpha)

# RR vs BL
rr_bl_p, rr_bl_CI = correlated_ttest(rr_bl, rho, alpha=alpha)

print('\nNN vs RR:', np.round(nn_rr_p,3), np.round(nn_rr_CI,2))
print('NN vs BL:', np.round(nn_bl_p,3), np.round(nn_bl_CI,2))
print('RR vs BL:', np.round(rr_bl_p,3), np.round(rr_bl_CI,2))


# Taken from https://piazza.com/class/k66atrohlm63kt?cid=183
# You can use both the CI and the p value to infer about the result. From
# statistics, if the calculated p-value is less than the alpha value, the
# zero hypothesis is rejected. That is, there is sufficient evidence to say
# that model A and B do not have the same performance (one is better than the
# other). The confidence interval is calculated using the alpha value. For
# alpha=0.05, the CI constitutes the interval in which we are 95% certain the
# true difference between model A and B lie. In the above example, the CI
# includes the value 0. A value of zero indicates that zA=zB (no difference).
# For there to be a difference, the CI must not include 0. If "z" describes
# accuracy, then zA is better than zB if the confidence interval is all positive.

# Result: We reject the null hypothesis that NN and BL have the same performance


#%% Clean up this mess

del i, j, K1, K2, y_bl_pred, lam, N, max_iter
del par_index, train_index, val_index, test_index
del min_error_nn_index, min_error_rr_index, n_replicates
del hidden_units, lambdas






