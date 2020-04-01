


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


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



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size = 0.25, random_state = 0)


from sklearn.model_selection import train_test_split
Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size = 0.25, random_state = 0)

# Applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
Xc_train = lda.fit_transform(Xc_train, yc_train)
Xc_test = lda.transform(Xc_test)

#----------------------Regression Models-------------------------------------

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression(n_jobs=-1)
lin_reg.fit(Xr_train, yr_train)

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
tree_regressor = DecisionTreeRegressor(random_state = 0)
tree_regressor.fit(Xr_train, yr_train)


# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(Xr_train)
poly_reg.fit(X_poly, yr_train)
lin_reg_2 = LinearRegression(n_jobs=-1)
lin_reg_2.fit(X_poly, yr_train)

#----------------------Classfication Models-----------------------------------

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
log_classifier = LogisticRegression(C=1, n_jobs=-1, random_state = 0)
log_classifier.fit(Xc_train, yc_train)


# Fitting  SVM to the Training set
from sklearn.svm import SVC
SVM_classifier = SVC(C=7, kernel = 'rbf',gamma=0.1, random_state = 0)
SVM_classifier.fit(Xc_train, yc_train)

# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
xgb_classifier = XGBClassifier(learning_rate= 0.001, n_jobs=-1, random_state = 0)
xgb_classifier.fit(Xc_train, yc_train)




#-----------------------------------------------------------
#PREDICTIONS



# Predicting a new result with Linear Regression
lin_pred = lin_reg.predict(Xr_test)

# Predicting a new result with Polynomial Regression
poly_pred= lin_reg_2.predict(poly_reg.fit_transform(Xr_test))


# Predicting the Test set results
log_pred = log_classifier.predict(Xc_test)

# Predicting the Test set results
xgb_pred = xgb_classifier.predict(Xc_test)


#Predicting the Test set results
SVM_pred = SVM_classifier.predict(Xc_test)





# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_log = confusion_matrix(yc_test, log_pred)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_xgb = confusion_matrix(yc_test, xgb_pred)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_SVM = confusion_matrix(yc_test, SVM_pred)

#-----------------------------------------------------------
#VALIDATION




# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
lin_reg_accuracies = cross_val_score(estimator = lin_reg, X = Xr_train, y = yr_train, cv = 10, n_jobs = -1)



# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
poly_reg_accuracies = cross_val_score(estimator = lin_reg_2, X = Xr_train, y = yr_train, cv = 10, n_jobs = -1)



# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
tree_accuracies = cross_val_score(estimator = tree_regressor, X = Xr_train, y = yr_train, cv = 10, n_jobs = -1)



# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
log_accuracies = cross_val_score(estimator = log_classifier, X = Xc_train, y = yc_train, cv = 10, n_jobs = -1)


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
SVM_accuracies = cross_val_score(estimator = SVM_classifier, X = Xc_train, y = yc_train, cv = 10, n_jobs = -1)



# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
xgb_accuracies = cross_val_score(estimator = xgb_classifier, X = Xc_train, y = yc_train, cv = 10, n_jobs = -1)




#-----------------------------------------------------------
#MODEL SELECTION





# Applying Grid Search to find the best model and the best parameters
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



from sklearn.model_selection import GridSearchCV
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





model_names = ['Linear Regression', 'Polynomial Regression', 'Decision Trees Regression', 'Logistic Reg', 'SVM', 'XGBoost']

accuracies_means = [lin_reg_accuracies.mean(), poly_reg_accuracies.mean(), tree_accuracies.mean(), log_accuracies.mean(), SVM_accuracies.mean(), xgb_accuracies.mean()]

accuracies_stds = [lin_reg_accuracies.std(), poly_reg_accuracies.std(), tree_accuracies.std(), log_accuracies.std(), SVM_accuracies.std(), xgb_accuracies.std()]
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
    print( round(accuracies_means[md],3),"\t", round(accuracies_stds[md],3),"\t", model_names[md],"\n")
    i+=1
for m in range(i,len(model_names)):
    print( round(accuracies_means[m],3),"\t", round(accuracies_stds[m],3),"\t", model_names[m],"\n")
            
 
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
print('MAE:', metrics.mean_absolute_error(yr_test, lin_pred))
print('MSE:', metrics.mean_squared_error(yr_test, lin_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(yr_test, lin_pred)))


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



