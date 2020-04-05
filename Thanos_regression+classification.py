


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from matplotlib.pylab import figure, plot, xlabel, ylabel, legend, show, boxplot, subplot, title, clim
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
regression_attribute_names=regression_heart_data.columns

scaler_reg = StandardScaler()
scaler_reg.fit(regression_heart_data)
Xr = scaler_reg.transform(regression_heart_data)   # What about y?s

# Non-standardized data
Xns = binary_heart_data.to_numpy()

N, M = Xr.shape

# Clean up variables
del scaler_binary, scaler_reg, fam_history, fh, heart_data, unique_hist, historyDict


#---------------------------------------------
#MODEL TRAINING

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size = 0.25, random_state = 0)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size = 0.25, random_state = 0)


# Applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
Xc_train_LDA = lda.fit_transform(Xc_train, yc_train)
Xc_test_LDA = lda.transform(Xc_test)


#----------------------Regression Models-------------------------------------






#PROJECT 2, Regression, PART a, points 1-2, --------------------------------

#INSERT A REGULARIZATION TERM IN EACH FOLD TO WORK PROPERLY, OTHERWISE YOU GET THE SAME RESULT IN EACH FOLD
#No feature selection

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn import model_selection
# K-fold crossvalidation
K = 10
CV = model_selection.KFold(n_splits=K,shuffle=True)

lc = np.arange(0.025,1.26,0.025)
poly = np.logspace(-3, 4, 50)
# Initialize variables
Error_train_lin = np.empty((len(lc),K))
Error_test_lin = np.empty((len(lc),K))

w=0
for train_index, test_index in CV.split(Xr):
    print('Computing logistic CV fold: {0}/{1}..'.format(w+1,K))

    # extract training and test set for current CV fold
    Xr_train_KFold_lin, yr_train_KFold_lin = Xr[train_index,:], yr[train_index]
    Xr_test_KFold_lin, yr_test_KFold_lin = Xr[test_index,:], yr[test_index]
    for d, f in enumerate(lc):
        
        # Fitting linear Regression model to the Training set
        lin_reg = LinearRegression( n_jobs=-1)
        lin_reg.fit(Xr_train_KFold_lin, yr_train_KFold_lin.ravel())
        lin_test_pred = lin_reg.predict(Xr_test_KFold_lin)
        lin_train_pred = lin_reg.predict(Xr_train_KFold_lin)
        misclass_rate_test = np.square(yr_test_KFold_lin-lin_test_pred).sum()/yr_test_KFold_lin.shape[0]
        misclass_rate_train = np.square(yr_train_KFold_lin-lin_train_pred).sum()/yr_train_KFold_lin.shape[0]
        Error_test_lin[d,w], Error_train_lin[d,w] = misclass_rate_test, misclass_rate_train
        
        # Fitting Polynomial Regression model to the Training set
        '''
        poly_reg = PolynomialFeatures(degree = 3, order='F')
        X_poly = poly_reg.fit_transform(Xr_train)
        poly_reg.fit(X_poly, yr_train)
        lin_reg_2 = LinearRegression(n_jobs=-1)
        lin_reg_2.fit(X_poly, yr_train)
        '''
        
        
    w+=1
#penalty='elasticnet', , solver='saga',l1_ratio=ratio



f = figure()
boxplot(Error_test_lin.T)
xlabel('Complexity: Regularization Factor')
ylabel('Test error across CV folds, K={0})'.format(K))
f = figure()
plot(lc, np.sqrt(Error_train_lin.mean(1)))
plot(lc, np.sqrt(Error_test_lin.mean(1)))
xlabel('Complexity: Regularization Factor')
ylabel('Error (RMSE, CV K={0})'.format(K))
legend(['Error_train','Error_test'])
    
show() 
 


#MOVING ON WITH THE BEST TUNED REGULARIZATION FACTOR('C')




#PROJECT 2, Regression, PART b, 1st point--------------------------------



## Linear regression model with nested cros validation AND feature selection

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
# Outer loop
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
    #Inner loop
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



#FITTING THE BEST SET OF FEATURES TO THE MODELS TO BE COMPARED






#----------------------Classfication Models-----------------------------------




'''
# Fitting  SVM to the Training set C=7 kerner='rbf', gamma=0.1 is the current optimal for cross_val_score
from sklearn.svm import SVC
SVM_classifier = SVC(C=7, kernel = 'linear',gamma=0.1, random_state = 0)
SVM_classifier.fit(Xc_train, yc_train)

 '''
 
#PROJECT 2, Classification, points 1-2-3 ----------------------

#LOGISTIC REGRESSION MODEL 

#Hyper-parameter training and selection


from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# K-fold crossvalidation
K = 10
CV = model_selection.KFold(n_splits=K,shuffle=True)

# Initialize variables
Error_train_base = np.empty((K,1))
Error_test_base = np.empty((K,1))
Error_train_GRID_log = np.empty((K,1))
Error_test_GRID_log = np.empty((K,1))
Error_train_RF = np.empty((K,1))
Error_test_RF = np.empty((K,1))

k=0
# Outer loop
for train_index, test_index in CV.split(Xr):
    
    # extract training and test set for current CV fold
    Xc_train_GRID_log = Xc[train_index,:]
    yc_train_GRID_log = yc[train_index]
    Xc_test_GRID_log = Xc[test_index,:]
    yc_test_GRID_log = yc[test_index]
    
    #Baseline Logistic Regression model with LDA transformation
    
    log_classifier_base = LogisticRegression( C=1,  n_jobs=-1, random_state = 0)
    log_classifier_base.fit(Xc_train_GRID_log, yc_train_GRID_log)
    
    # Applying LDA

    lda = LDA(n_components = 2)
    Xc_train_LDA = lda.fit_transform(Xc_train_GRID_log, yc_train_GRID_log)
    Xc_test_LDA = lda.transform(Xc_test_GRID_log)
   
    log_classifier = LogisticRegression( C=1,  n_jobs=-1, random_state = 0)
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
    print('Cross validation fold {0}/{1}'.format(k+1,K))
    print('Train Error: {0}'.format(Error_train_GRID_log[k]))
    print('Test Error: {0}'.format(Error_test_GRID_log[k]))
    
    
    
    # Fitting Random Forest Regression to the dataset
    
    rand_forest = RandomForestRegressor(n_estimators = 290, max_depth=12, min_samples_split= 15,min_samples_leaf=2, max_features="auto",bootstrap=True, random_state = 0, n_jobs= -1)
    rand_forest.fit(Xc_train_GRID_log, yc_train_GRID_log)
    #Error calculation for the Random Forest model
    misclass_rate_test_RF = sum(rand_forest.predict(Xc_test_GRID_log) != yc_test_GRID_log) / float(len(rand_forest.predict(Xc_test_GRID_log)))
    misclass_rate_train_RF = sum(rand_forest.predict(Xc_train_GRID_log) != yc_train_GRID_log) / float(len(rand_forest.predict(Xc_train_GRID_log)))
    Error_test_RF[k], Error_train_RF[k] = misclass_rate_test_RF, misclass_rate_train_RF
    
    k+=1
    
print("For baseline Logistisc Regression: ")
print('Train Error Accuracy({0}Kforld): {1}\n'.format(K,Error_train_base.T.mean(1)))
print('Test Error Accuracy({0}Kforld): {1}\n'.format(K,Error_test_base.T.mean(1)))   
    
print("For optimized Logistisc Regression: ")
print('Train Error Accuracy({0}Kforld): {1}\n'.format(K,Error_train_GRID_log.T.mean(1)))
print('Test Error Accuracy({0}Kforld): {1}\n'.format(K,Error_test_GRID_log.T.mean(1)))

print("For Random Forests classification: ")
print('Train Error Accuracy({0}Kforld): {1}\n'.format(K,Error_train_RF.T.mean(1)))
print('Test Error Accuracy({0}Kforld): {1}\n'.format(K,Error_test_RF.T.mean(1)))
#mean_train = Error_train_GRID_log.mean(1)
#mean_test = Error_test_GRID_log.mean(1)
C=[0.00000001,0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10]
import matplotlib.pyplot as plt
plt.semilogx(C, Error_train_base*100)
plt.semilogx(C, Error_test_base*100)
plt.semilogx(C, Error_train_GRID_log*100)
plt.semilogx(C, Error_test_GRID_log*100)
plt.semilogx(C, Error_train_RF*100)
plt.semilogx(C, Error_test_RF*100)
xlabel('Regularization factor')
ylabel('Error (%), CV K={0}'.format(K))
legend(['Error_train','Error_test'],loc=0)


plt.show()


#0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100


#Nested CV of the tested models

from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection

# K-fold crossvalidation
K = 10
CV = model_selection.KFold(n_splits=K,shuffle=True)

lc = np.arange(0.025,1.25,0.125)
tc = np.arange(10,500,50)



# Initialize variables
Error_train_log = np.empty((len(lc),K))
Error_test_log = np.empty((len(lc),K))
Error_train_log_intercept = np.empty((len(lc),K))
Error_test_log_intercept = np.empty((len(lc),K))
Error_train = np.empty((len(tc),K))
Error_test = np.empty((len(tc),K))

ratio=0.1
j=0
k=0
for train_index, test_index in CV.split(Xc):
    print('Computing logistic CV fold: {0}/{1}..'.format(j+1,K))

    # extract training and test set for current CV fold
    Xc_train_KFold_log, yc_train_KFold_log = Xc[train_index,:], yc[train_index]
    Xc_test_KFold_log, yc_test_KFold_log = Xc[test_index,:], yc[test_index]
    for a, c in enumerate(lc):
        
        # Fitting the baseline Logistic Regression model to the Training set
        log_classifier_intercept = LogisticRegression(fit_intercept=True,  n_jobs=-1, random_state = 0)
        log_classifier_intercept.fit(Xc_train_KFold_log, yc_train_KFold_log.ravel())
        log_intercept_test_pred = log_classifier_intercept.predict(Xc_test_KFold_log)
        log_intercept_train_pred = log_classifier_intercept.predict(Xc_train_KFold_log)
        misclass_rate_test_intercept = np.square(yc_test_KFold_log-log_intercept_test_pred).sum()/yc_test_KFold_log.shape[0]
        misclass_rate_train_intercept = np.square(yc_train_KFold_log-log_intercept_train_pred).sum()/yc_train_KFold_log.shape[0]
        Error_test_log_intercept[a,j], Error_train_log_intercept[a,j] = misclass_rate_test_intercept, misclass_rate_train_intercept
        
        
        # Fitting Logistic Regression model to the Training set
        log_classifier = LogisticRegression( C=c, n_jobs=-1, random_state = 0)
        log_classifier.fit(Xc_train_KFold_log, yc_train_KFold_log.ravel())
        log_test_pred = log_classifier.predict(Xc_test_KFold_log)
        log_train_pred = log_classifier.predict(Xc_train_KFold_log)
        misclass_rate_test = np.square(yc_test_KFold_log-log_test_pred).sum()/yc_test_KFold_log.shape[0]
        misclass_rate_train = np.square(yc_train_KFold_log-log_train_pred).sum()/yc_train_KFold_log.shape[0]
        Error_test_log[a,j], Error_train_log[a,j] = misclass_rate_test, misclass_rate_train
    print('Computing RF CV fold: {0}/{1}..'.format(k+1,K))    
    for i, t in enumerate(tc):
        # Fitting Random Forest model to the Training set
        randc_forest = RandomForestClassifier(n_estimators = t, criterion = 'entropy', max_depth=3, min_samples_leaf=i+1, min_samples_split=i+2, max_features="auto", bootstrap=False, random_state = 0, n_jobs= -1)
        randc_forest.fit(Xc_train_KFold_log, yc_train_KFold_log.ravel())  

        rfc_test_pred = randc_forest.predict(Xc_test_KFold_log)
        rfc_train_pred = randc_forest.predict(Xc_train_KFold_log)
        misclass_rate_test = np.square(yc_test_KFold_log-rfc_test_pred).sum()/yc_test_KFold_log.shape[0]
        misclass_rate_train = np.square(yc_train_KFold_log-rfc_train_pred).sum()/yc_train_KFold_log.shape[0]
        Error_test[i,k], Error_train[i,k] = misclass_rate_test, misclass_rate_train
    k+=1
    j+=1
    ratio+=0.1
#penalty='elasticnet', , solver='saga',l1_ratio=ratio
f = figure()
boxplot(Error_test_log.T)
xlabel('Log Complexity: Regularization Factor')
ylabel('Test error across CV folds, K={0})'.format(K))
f = figure()
plot(lc, np.sqrt(Error_train_log.mean(1)))
plot(lc, np.sqrt(Error_test_log.mean(1)))
xlabel('Log Complexity: Regularization Factor')
ylabel('Error (RMSE, CV K={0})'.format(K))
legend(['Error_train','Error_test'])
f = figure()
plot(lc, np.sqrt(Error_train_log_intercept.mean(1)))
plot(lc, np.sqrt(Error_test_log_intercept.mean(1)))
xlabel('Log Complexity: Regularization Factor')
ylabel('Error (RMSE, CV K={0})'.format(K))
legend(['Error_train','Error_test'])
f = figure()
boxplot(Error_test.T)
xlabel('RF Complexity: Estimators-min_samples leaf+Split')
ylabel('Test error across CV folds, K={0})'.format(K))
f = figure()
plot(tc, np.sqrt(Error_train.mean(1)))
plot(tc, np.sqrt(Error_test.mean(1)))
xlabel('RF Complexity: Estimators-min_samples leaf+Split')
ylabel('Error (RMSE, CV K={0})'.format(K))
legend(['Error_train','Error_test'])
        
show() 
 




import matplotlib.pyplot as plt
plt.plot(tc,np.sqrt(Error_test.mean(1)), marker='o', markerfacecolor='blue', markersize=7, color='skyblue', linewidth=3,label='Random forests' )
plt.plot(tc, np.sqrt(Error_test_log.mean(1)), marker='', color='olive', linewidth=3, label= 'Logistic Regression')
plt.plot(tc,np.sqrt(Error_test_log_intercept.mean(1)), marker='', color='olive', linewidth=3, linestyle='dashed', label="Baseline")
xlabel('Complexity')
ylabel('RMSE')
plt.legend(loc=0)

plt.show()










'''


# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
xgb_classifier = XGBClassifier(learning_rate= 0.001, n_jobs=-1, random_state = 0)
xgb_classifier.fit(Xc_train, yc_train)


# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
randc_forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0, n_jobs= -1)
randc_forest.fit(Xc_train, yc_train)

#-----------------------------------------------------------
#PREDICTIONS



# Predicting a new result with Linear Regression
lin_train_pred = lin_reg.predict(Xr_train)
lin_test_pred = lin_reg.predict(Xr_test)

# Predicting a new result with Polynomial Regression
poly_train_pred= lin_reg_2.predict(poly_reg.fit_transform(Xr_train))
poly_test_pred= lin_reg_2.predict(poly_reg.fit_transform(Xr_test))

# Predicting a new result with Random Forests
rf_train_pred = rand_forest.predict(Xr_train)
rf_train_pred = rand_forest.predict(Xr_test)

# Predicting the Test set results
log_train_pred = log_classifier.predict(Xc_train)
log_test_pred = log_classifier.predict(Xc_test)

# Predicting the Test set results
xgb_train_pred = xgb_classifier.predict(Xc_train)
xgb_test_pred = xgb_classifier.predict(Xc_test)

#Predicting the Test set results
SVM_train_pred = SVM_classifier.predict(Xc_train)
SVM_test_pred = SVM_classifier.predict(Xc_test)

rfc_train_pred =randc_forest.predict(Xc_train)
rfc_test_pred =randc_forest.predict(Xc_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_log = confusion_matrix(yc_test, log_test_pred)


# Making the Confusion Matrix

cm_xgb = confusion_matrix(yc_test, xgb_test_pred)

# Making the Confusion Matrix

cm_SVM = confusion_matrix(yc_test, SVM_test_pred)
'''
#-----------------------------------------------------------
#ERROR CALCULATION





#(Classification)

'''
tc = np.arange(1, 11, 1)

# Initialize variables
Error_train = np.empty((len(tc),1))
Error_test = np.empty((len(tc),1))


for i, t in enumerate(tc):
    SVM_test_pred = np.asarray(SVM_classifier.predict(Xc_test),dtype=int)
    SVM_train_pred = np.asarray(SVM_classifier.predict(Xc_train), dtype=int)
    SVM_misclass_rate_test = sum(SVM_test_pred != yc_test) / len(SVM_test_pred)
    SVM_misclass_rate_train = sum(SVM_train_pred != yc_train) /len(SVM_train_pred)
    Error_test[i], Error_train[i] = SVM_misclass_rate_test, SVM_misclass_rate_train

f = figure()
plot(tc, Error_train*100)
plot(tc, Error_test*100)
xlabel('Model complexity (max tree depth)')
ylabel('Error (%)')
legend(['Error_train','Error_test'])
    
show()    

#-----------------------------------------------------------
#VALIDATION




# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
lin_reg_accuracies = cross_val_score(estimator = lin_reg, X = Xr_train, y = yr_train, cv = 10, n_jobs = -1)



# Applying k-Fold Cross Validation

poly_reg_accuracies = cross_val_score(estimator = lin_reg_2, X = Xr_train, y = yr_train, cv = 10, n_jobs = -1)



# Applying k-Fold Cross Validation

rf_accuracies = cross_val_score(estimator = rand_forest, X = Xr_train, y = yr_train, cv = 10, n_jobs = -1)



# Applying k-Fold Cross Validation

log_accuracies = cross_val_score(estimator = log_classifier, X = Xc_train, y = yc_train, cv = 10, n_jobs = -1)


# Applying k-Fold Cross Validation

SVM_accuracies = cross_val_score(estimator = SVM_classifier, X = Xc_train, y = yc_train, cv = 10, n_jobs = -1)



# Applying k-Fold Cross Validation

xgb_accuracies = cross_val_score(estimator = xgb_classifier, X = Xc_train, y = yc_train, cv = 10, n_jobs = -1)

# Applying k-Fold Cross Validation

rfc_accuracies = cross_val_score(estimator = randc_forest, X = Xc_train, y = yc_train, cv = 10, n_jobs = -1)


#-----------------------------------------------------------
#MODEL SELECTION

#RANDOM FORESTS MODEL

#Regression


# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
rand_forest = RandomForestRegressor(n_estimators = 290, max_depth=12, min_samples_split= 15,min_samples_leaf=2, max_features="auto",bootstrap=True, random_state = 0, n_jobs= -1)
rand_forest.fit(Xr_train, yr_train)


#Classification
 

from sklearn.ensemble import RandomForestClassifier    
# K-fold crossvalidation
K = 10
CV = model_selection.KFold(n_splits=K,shuffle=True)

tc = np.arange(10,500,50)

# Initialize variables
Error_train = np.empty((len(tc),K))
Error_test = np.empty((len(tc),K))

k=0
for train_index, test_index in CV.split(Xc):
    print('Computing RF CV fold: {0}/{1}..'.format(k+1,K))

    # extract training and test set for current CV fold
    Xc_train_KFold, yc_train_KFold = Xc[train_index,:], yc[train_index]
    Xc_test_KFold, yc_test_KFold = Xc[test_index,:], yc[test_index]
    for i, t in enumerate(tc):
        randc_forest = RandomForestClassifier(n_estimators = t, criterion = 'entropy', max_depth=3, min_samples_leaf=i+1, min_samples_split=i+2, max_features="auto", bootstrap=False, random_state = 0, n_jobs= -1)
        randc_forest.fit(Xc_train_KFold, yc_train_KFold.ravel())  

        rfc_test_pred = randc_forest.predict(Xc_test_KFold)
        rfc_train_pred = randc_forest.predict(Xc_train_KFold)
        misclass_rate_test = np.square(yc_test_KFold-rfc_test_pred).sum()/yc_test_KFold.shape[0]
        misclass_rate_train = np.square(yc_train_KFold-rfc_train_pred).sum()/yc_train_KFold.shape[0]
        Error_test[i,k], Error_train[i,k] = misclass_rate_test, misclass_rate_train
    k+=1

f = figure()
boxplot(Error_test.T)
xlabel('RF Complexity: Estimators-min_samples leaf+Split')
ylabel('Test error across CV folds, K={0})'.format(K))
f = figure()
plot(tc, np.sqrt(Error_train.mean(1)))
plot(tc, np.sqrt(Error_test.mean(1)))
xlabel('RF Complexity: Estimators-min_samples leaf+Split')
ylabel('Error (RMSE, CV K={0})'.format(K))
legend(['Error_train','Error_test'])
    
show()




# Applying Grid Search to find the best model and the best parameters for classification Log Regression
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1,2,3,4,5]}]
grid_search = GridSearchCV(estimator = log_classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search_log = grid_search.fit(Xc_train, yc_train)
best_accuracy_log = grid_search.best_score_
best_parameters_log = grid_search.best_params_

# Applying Grid Search to find the best model and the best parameters

#Grid search for regression Random Forests 
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

#Grid search for classification SVM

parameters = [{'C': [1,2,3], 'kernel': ['linear']},
              {'C': [6,7,8,9,10,11,12,13,14,15,16,17,18,19], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = SVM_classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search_SVM = grid_search.fit(Xc_train, yc_train)
best_accuracy_SVM = grid_search.best_score_
best_parameters_SVM = grid_search.best_params_





model_names = ['Linear Regression', 'Polynomial Regression', 'Random Forests Regression', 'Logistic Reg', 'SVM', 'XGBoost']

accuracies_means = [lin_reg_accuracies.mean(), poly_reg_accuracies.mean(), rf_accuracies.mean(), log_accuracies.mean(), SVM_accuracies.mean(), xgb_accuracies.mean()]

accuracies_stds = [lin_reg_accuracies.std(), poly_reg_accuracies.std(), rf_accuracies.std(), log_accuracies.std(), SVM_accuracies.std(), xgb_accuracies.std()]

print()
print()
print("The mean and variance of accuracies for the following models are: ")
print()
print('------------Regression-----------')
print()
i=0
for md in range(0,len(model_names)):
    while 'Regression' in model_names[md]:
        break
    else:
        print('----------Classification------------')
        print()
        break
    print( round(accuracies_means[md],3),"\t", round(accuracies_stds[md],3),"\t", model_names[md])
    print("Accuracy: %0.2f (+/- %0.2f)" % (accuracies_means[md], accuracies_stds[md] * 2))
    i+=1
    print()
for m in range(i,len(model_names)):
    print( round(accuracies_means[m],3),"\t", round(accuracies_stds[m],3),"\t", model_names[m])
    print("Accuracy: %0.2f (+/- %0.2f)" % (accuracies_means[md], accuracies_stds[md] * 2))
    print()
 
#------------------------------------------------------------------------------------
#REGRESSIION METRICS



print()
print("Regression error metrics:")
print()
#MAE is the easiest to understand, because it's the average error.
#MSE is more popular than MAE, because MSE "punishes" larger errors, which tends to be useful in the real world.
#RMSE is even more popular than MSE, because RMSE is interpretable in the "y" units.
#All of these are loss functions, because we want to minimize them.
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(yr_test, lin_test_pred))
print('MSE:', metrics.mean_squared_error(yr_test, lin_test_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(yr_test, lin_test_pred)))


#------------------------------------------------------------------------------------
#CLASSIFICATION METRICS



from sklearn.metrics import classification_report
print(classification_report(yr_test,SVM_pred))

'''


#%%---------------------------------------------------------
#VISUALIZATIONS


'''
# Visualising the Linear Regression results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X, lin_reg.predict(X_train), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, lin_reg_2.predict(poly_reg.fit_transform(X_train), color = 'blue'))
plt.title('Truth or Bluff Polynomial Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X_train), max(X_train), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01) )
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
'''

#-----------------------------------------------------------
#IDLE CODE


#Error_train[i,k] = np.square(yc_train_KFold-rfc_train_pred).sum()/yc_train_KFold.shape[0]
#Error_test[i,k] = np.square(yc_test_KFold-rfc_test_pred).sum()/yc_test_KFold.shape[0]

#sum(rfc_test_pred != yc_test_KFold) / float(len(rfc_test_pred))
#sum(rfc_train_pred != yc_train_KFold) /float(len(rfc_train_pred))
