
from load_data import X, y, binary_heart_data

import numpy as np
import sklearn.linear_model as lm
from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, subplot, hist, show


# regress for adi/age & adi/obes ?
# adi/(age+obes+???) ? 
# adi/(all data) ?



# Prepare data
y_nu = np.array([X[:,3]]).T # adiposity
age = np.array([X[:,6]]).T
# obes = np.array([X[:,8]]).T


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




# f = figure()
# plot(obes,adi,'.')