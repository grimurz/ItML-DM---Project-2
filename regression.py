
from load_data import X, y, binary_heart_data

import numpy as np
import sklearn.linear_model as lm
from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, subplot, hist, show


# regress for adi/age & adi/(all data) ?


# Prepare data
X = X[np.argsort(X[:, 6])] # line plot gets otherwise messy

y_nu = np.array(X[:,3]).reshape(-1,1) # adiposity
age = np.array(X[:,6]).reshape(-1,1)

X = X[:,[0,1,2, 4,5,6,7,8]] # exclude adiposity

age_pow = age**2
age2 = np.asarray(np.bmat('age, age_pow'))



### TEST 1 ### adi/age

# Fit ordinary least squares regression model
model = lm.LinearRegression(fit_intercept=True)
model = model.fit(age,y_nu)

# Compute model output:
y_est = model.predict(age)
residual = y_est - y_nu

# Plot
f = figure()
plot(age,y_nu,'.')
plot(age,y_est,'-')

figure()
hist(residual,40)



### TEST 2 ### adi/(age,age^2)

# Fit ordinary least squares regression model
model = lm.LinearRegression(fit_intercept=True)
model = model.fit(age2,y_nu)

# Compute model output:
y_est = model.predict(age2)
residual = y_est - y_nu

# Plot
f = figure()
plot(age,y_nu,'.')
plot(age,y_est,'-')

figure()
hist(residual,40)



### TEST 3 ### adi/X

# Fit ordinary least squares regression model
model = lm.LinearRegression(fit_intercept=True)
model = model.fit(X,y_nu)

# Compute model output:
y_est = model.predict(X)
residual = y_est - y_nu

# Plot
f = figure()
plot(y_nu,y_est,'.')

figure()
hist(residual,40)




del model, f