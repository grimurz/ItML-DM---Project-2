
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
binary_heart_data = heart_data.copy()
binary_heart_data.famhist = fh
y = heart_data[['chd']].to_numpy().squeeze()
binary_heart_data.drop('row.names', axis=1, inplace=True)
binary_heart_data.drop('chd', axis=1, inplace=True)

# Data standardization: We scale our data so that each feature has
# a single unit of variance.
scaler = StandardScaler()
scaler.fit(binary_heart_data)
X = scaler.transform(binary_heart_data)

# Non-standardized data
Xns = binary_heart_data.to_numpy()

# Clean up variables
del scaler, fam_history, fh, heart_data, unique_hist
# del historyDict