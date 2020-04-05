

#OPTIMIZING LOG REG MODEL WITH Kfold CV
#FIND A WAY TO TEST EACH GRID SEARCH FOLD WITH THE TEST SET



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






from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# K-fold crossvalidation
K = 10
CV = model_selection.KFold(n_splits=K,shuffle=True)

# Initialize variables
Error_train_GRID_log = np.empty((K,1))
Error_test_GRID_log = np.empty((K,1))

k=0
# Outer loop
for train_index, test_index in CV.split(Xr):
    
    # extract training and test set for current CV fold
    Xc_train_GRID_log = Xc[train_index,:]
    yc_train_GRID_log = yc[train_index]
    Xc_test_GRID_log = Xc[test_index,:]
    yc_test_GRID_log = yc[test_index]
    
    # Applying LDA

    lda = LDA(n_components = 2)
    Xc_train_LDA = lda.fit_transform(Xc_train_GRID_log, yc_train_GRID_log)
    Xc_test_LDA = lda.transform(Xc_test_GRID_log)
   
    log_classifier = LogisticRegression( C=1,  n_jobs=-1, random_state = 0)
    log_classifier.fit(Xc_train_LDA, yc_train_GRID_log)

    #Grid search for classification SVM

    parameters = [{'C': [0.00000001,0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10]}]
    grid_search = GridSearchCV(estimator = log_classifier,
                               param_grid = parameters,
                               scoring = 'neg_log_loss',
                               cv = 10,
                               n_jobs = -1)
    grid_search= grid_search.fit(Xc_train_LDA, yc_train_GRID_log)
    
    misclass_rate_test = sum(grid_search.predict(Xc_test_LDA) != yc_test_GRID_log) / float(len(grid_search.predict(Xc_test_LDA)))
    misclass_rate_train = sum(grid_search.predict(Xc_train_LDA) != yc_train_GRID_log) / float(len(grid_search.predict(Xc_train_LDA)))
    Error_test_GRID_log[k], Error_train_GRID_log[k] = misclass_rate_test, misclass_rate_train
    best_accuracy_log = grid_search.best_score_
    best_parameters_log = grid_search.best_params_
    print('Cross validation fold {0}/{1}'.format(k+1,K))
    print('Train Error: {0}'.format(Error_train_GRID_log[k]))
    print('Test Error: {0}'.format(Error_test_GRID_log[k]))
    k+=1
print('Train Error Accuracy({0}Kforld): {1}\n'.format(K,Error_train_GRID_log.T.mean(1)))
print('Test Error Accuracy({0}Kforld): {1}\n'.format(K,Error_test_GRID_log.T.mean(1)))
#mean_train = Error_train_GRID_log.mean(1)
#mean_test = Error_test_GRID_log.mean(1)
C=[0.00000001,0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10]
import matplotlib.pyplot as plt
plt.semilogx(C, Error_train_GRID_log*100)
plt.semilogx(C, Error_test_GRID_log*100)
xlabel('Regularization factor')
ylabel('Error (%), CV K={0}'.format(K))
legend(['Error_train','Error_test'],loc=0)


plt.show()


#0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100





'''
#This works for one-layer regression errors

Error_train_GRID_log[k] = np.sqrt(np.square(yc_train_GRID_log-log_classifier.predict(Xc_train_LDA)).sum()/yc_train_GRID_log.shape[0])
Error_test_GRID_log[k] = np.sqrt(np.square(yc_test_GRID_log-log_classifier.predict(Xc_test_LDA)).sum()/yc_test_GRID_log.shape[0])

plt.semilogx(C,Error_train_GRID_log.mean(1), marker='o', markerfacecolor='blue', markersize=7, color='skyblue', linewidth=3,label='Train error' )
plt.semilogx(C, Error_test_GRID_log.mean(1), marker='', color='olive', linewidth=3, label= 'Test error')
#plt.plot(tc,Error_test_GRID_log.mean(1), marker='', color='olive', linewidth=3, linestyle='dashed', label="Baseline")
xlabel('Regularization Factor')
ylabel('RMSE')
plt.legend(loc=0)
'''




