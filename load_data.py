
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

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
bin_data = heart_data.copy()
bin_data.famhist = fh

y_r = heart_data[['age']].to_numpy().squeeze()
y_c = heart_data[['chd']].to_numpy().squeeze()

bin_data.drop('row.names', axis=1, inplace=True)

df_r = bin_data.copy()
df_c = bin_data.copy()

df_r.drop('age', axis=1, inplace=True)
df_c.drop('chd', axis=1, inplace=True)


# Data standardization: We scale our data so that each feature has
# a single unit of variance.
scaler = StandardScaler()
scaler.fit(df_r)
scaler.fit(df_c)
X_r = scaler.transform(df_r)
X_c = scaler.transform(df_c)

# Non-standardized data
Xns_r = df_r.to_numpy()
Xns_c = df_c.to_numpy()

# Clean up variables
del scaler, fam_history, fh, heart_data, unique_hist, historyDict, df_r, df_c