import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from matplotlib.pylab import figure, plot, xlabel, ylabel, legend, show, boxplot, subplot, title, clim
from sklearn import model_selection, metrics
from toolbox_02450 import feature_selector_lr, bmplot
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LinearRegression
from sklearn import model_selection, linear_model

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
binary_attribute_names = binary_heart_data.columns

scaler_reg = StandardScaler()
scaler_reg.fit(regression_heart_data)
Xr = scaler_reg.transform(regression_heart_data)   # What about y?s

# Non-standardized data
Xns = binary_heart_data.to_numpy()

N, M = Xr.shape








def gcm_validate(X,y,cvf=10):
    ''' Validate linear regression model using 'cvf'-fold cross validation.
        The loss function computed as mean squared error on validation set (MSE).
        Function returns MSE averaged over 'cvf' folds.

        Parameters:
        X       training data set
        y       vector of values
        cvf     number of crossvalidation folds        
    '''
    y = y.squeeze()
    CV = model_selection.KFold(n_splits=cvf, shuffle=True)
    validation_error=np.empty(cvf)
    f=0
    for train_index, test_index in CV.split(X):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        
        logistic = LogisticRegression( C=1, n_jobs=-1, random_state = 0)
        logistic.fit(X_train, y_train)
        log_test_pred = logistic.predict(X_test)
        bool_pred =np.array(log_test_pred != y_test).astype(int)
        count_zeros =np.count_nonzero(bool_pred==1)
        validation_error[f]= count_zeros/y_test.shape[0]
        f=f+1
    return validation_error.mean(0)        

def glm_validate(X,y,cvf=10):
    ''' Validate linear regression model using 'cvf'-fold cross validation.
        The loss function computed as mean squared error on validation set (MSE).
        Function returns MSE averaged over 'cvf' folds.

        Parameters:
        X       training data set
        y       vector of values
        cvf     number of crossvalidation folds        
    '''
    y = y.squeeze()
    CV = model_selection.KFold(n_splits=cvf, shuffle=True)
    validation_error=np.empty(cvf)
    f=0
    for train_index, test_index in CV.split(X):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        m = linear_model.LinearRegression(fit_intercept=True).fit(X_train, y_train)
        validation_error[f] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]
        f=f+1
    return validation_error.mean()


def feature_classifier(X,y,cvf=10,features_record=None,loss_record=None,display=''):
    ''' Function performs feature selection for linear regression model using
        'cvf'-fold cross validation. The process starts with empty set of
        features, and in every recurrent step one feature is added to the set
        (the feature that minimized loss function in cross-validation.)

        Parameters:
        X       training data set
        y       vector of values
        cvf     number of crossvalidation folds

        Returns:
        selected_features   indices of optimal set of features
        features_record     boolean matrix where columns correspond to features
                            selected in subsequent steps
        loss_record         vector with cv errors in subsequent steps
        
        Example:
        selected_features, features_record, loss_record = ...
            feature_selector_lr(X_train, y_train, cvf=10)
            
    ''' 
    y = y.squeeze() #ÆNDRING JLH #9/3
    # first iteration error corresponds to no-feature estimator
    if loss_record is None:
        loss_record = np.array([np.square(y-y.mean()).sum()/y.shape[0]])
    if features_record is None:
        features_record = np.zeros((X.shape[1],1))

    # Add one feature at a time to find the most significant one.
    # Include only features not added before.
    selected_features = features_record[:,-1].nonzero()[0]
    min_loss = loss_record[-1]
    if display is 'verbose':
        print(min_loss)
    best_feature = False
    for feature in range(0,X.shape[1]):
        if np.where(selected_features==feature)[0].size==0:
            trial_selected = np.concatenate((selected_features,np.array([feature])),0).astype(int)
            # validate selected features with linear regression and cross-validation:
            trial_loss = gcm_validate(X[:,trial_selected],y,cvf)
            if display is 'verbose':
                print(trial_loss)
            if trial_loss<min_loss:
                min_loss = trial_loss 
                best_feature = feature

    # If adding extra feature decreased the loss function, update records
    # and go to the next recursive step
    if best_feature is not False:
        features_record = np.concatenate((features_record, np.array([features_record[:,-1]]).T), 1)
        features_record[best_feature,-1]=1
        loss_record = np.concatenate((loss_record,np.array([min_loss])),0)
        selected_features, features_record, loss_record = feature_classifier(X,y,cvf,features_record,loss_record)
        
    # Return current records and terminate procedure
    return selected_features, features_record, loss_record


K = 10
CV = model_selection.KFold(n_splits=K,shuffle=True)
from sklearn.linear_model import LinearRegression
# Initialize variables
Features = np.zeros((M,K))
Error_train_logFS = np.empty((K,1))
Error_test_logFS = np.empty((K,1))
Error_train_rf = np.empty((K,1))
Error_test_rf = np.empty((K,1))
Error_train_fsLDA= np.empty((K,1))
Error_test_fsLDA= np.empty((K,1))

k=0
# Outer loop
for train_index, test_index in CV.split(Xc):
    
    # extract training and test set for current CV fold
    Xc_train = Xc[train_index,:]
    yc_train = yc[train_index]
    Xc_test = Xc[test_index,:]
    yc_test = yc[test_index]
    internal_cross_validation = 10
    
   
    # Compute  error with feature subset selection
    textout = ''
    selected_features, features_record, loss_record = feature_classifier(Xc_train, yc_train, internal_cross_validation,display=textout)
    #Inner loop
    Features[selected_features,k] = 1
    # .. alternatively you could use module sklearn.feature_selection
    if len(selected_features) is 0:
        print('No features were selected, i.e. the data (X) in the fold cannot describe the outcomes (y).' )
    else:
        
        
         # Fitting Random Forest model to the Training set
        randc_forestFS = RandomForestClassifier(n_estimators = 400, criterion = 'entropy', max_depth=3, min_samples_leaf=4, min_samples_split=9, max_features="auto", bootstrap=False, random_state = 0, n_jobs= -1)
        randc_forestFS.fit(Xc_train[:,selected_features], yc_train)  

        rfc_test_pred_fs = randc_forestFS.predict(Xc_test[:,selected_features])
        rfc_train_pred_fs = randc_forestFS.predict(Xc_train[:,selected_features])
        
        misclass_rate_test_rf = sum(rfc_test_pred_fs != yc_test) / float(len(rfc_test_pred_fs))
        misclass_rate_train_rf = sum(rfc_train_pred_fs != yc_train) / float(len(rfc_train_pred_fs))
        Error_test_rf[k], Error_train_rf[k] = misclass_rate_test_rf, misclass_rate_train_rf
        
        

        
        #Feature selection for log reg
        log_classifier = LogisticRegression(C=1,  n_jobs=-1, random_state = 0)
        log_classifier.fit(Xc_train[:,selected_features], yc_train)
        
        log_test_pred = log_classifier.predict(Xc_test[:,selected_features])
        log_train_pred = log_classifier.predict(Xc_train[:,selected_features])
        
        
        misclass_rate_test = sum(log_test_pred != yc_test) / float(len(log_test_pred))
        misclass_rate_train= sum(log_train_pred != yc_train) / float(len(log_train_pred))
        Error_test_logFS[k], Error_train_logFS[k] = misclass_rate_test, misclass_rate_train
        
        
      
        # Applying LDA
        lda = LDA(n_components = 2)
        Xc_train_LDA = lda.fit_transform(Xc_train[:,selected_features], yc_train)
        Xc_test_LDA = lda.transform(Xc_test[:,selected_features])
        #Retraining of the Logistic Regression model with selected features and LDA analysis on the selected features
        log_classifier_fsLDA = LogisticRegression(fit_intercept=True,  n_jobs=-1, random_state = 0)
        log_classifier_fsLDA.fit(Xc_train_LDA, yc_train)
        
        log_test_pred_fsLDA = log_classifier_fsLDA.predict(Xc_test_LDA)
        log_train_pred_fsLDA = log_classifier_fsLDA.predict(Xc_train_LDA)
        
        
        misclass_rate_test_fsLDA = sum(log_test_pred != yc_test) / float(len(log_test_pred))
        misclass_rate_train_fsLDA= sum(log_train_pred != yc_train) / float(len(log_train_pred))
        Error_test_fsLDA[k], Error_train_fsLDA[k] = misclass_rate_test_fsLDA, misclass_rate_train_fsLDA
        
        
        figure(k)
        subplot(1,2,1)
        plot(range(1,len(loss_record)), loss_record[1:])
        title('Classification model number: {0}'.format(k))
        xlabel('Iteration')
        ylabel('Error (crossvalidation)')    
        
        subplot(1,3,3)
        bmplot(binary_attribute_names, range(1,features_record.shape[1]), -features_record[:,1:])
        clim(-1.5,0)
        xlabel('Iteration')


    print('Cross validation fold {0}/{1}'.format(k+1,K))
    #print('Train indices: {0}'.format(train_index))
    #print('Test indices: {0}'.format(test_index))
    print('Features no: {0}\n'.format(selected_features.size))

    k+=1
'''
# Display results
print('\n')
print('Baseline regression without feature selection:\n')
print('- Training error: {0}'.format(Error_train_log_intercept.mean()))
print('- Test error:     {0}'.format(Error_test_log_intercept.mean()))


print('Logistic Regression without feature selection:\n')
print('- Training error: {0}'.format(Error_train_log.mean()))
print('- Test error:     {0}'.format(Error_test_log.mean()))


print('Logistic Regression with feature selection:\n')
print('- Training error: {0}'.format(Error_train_logFS.mean()))
print('- Test error:     {0}'.format(Error_test_logFS.mean()))


print('Random Forests without feature selection:\n')
print('- Training error: {0}'.format(Error_train_RF.mean()))
print('- Test error:     {0}'.format(Error_test_RF.mean()))

print('Random Forests with feature selection:\n')
print('- Training error: {0}'.format(Error_train_rf.mean()))
print('- Test error:     {0}'.format(Error_test_rf.mean()))
'''
figure(k)
subplot(1,3,2)
bmplot(binary_attribute_names, range(1,Features.shape[1]+1), -Features)
clim(-1.5,0)
xlabel('Crossvalidation fold')
ylabel('Attribute')
title('Classification')
show()




