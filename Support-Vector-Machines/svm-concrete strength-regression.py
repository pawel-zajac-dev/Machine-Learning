import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# get the data: https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength
df = pd.read_excel('Concrete_Data.xls')
df.columns = list(range(df.shape[1]))

X = df[[0,1,2,3,4,5,6,7]].values
Y = df[8].values

# split the data into train and test sets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33)

# scale the data
scaler = StandardScaler()
Xtrain = scaler.fit_transform(Xtrain)
Xtest = scaler.transform(Xtest)

target_scaler = StandardScaler()
Ytrain = target_scaler.fit_transform(Ytrain.reshape(-1, 1)).flatten()
Ytest = target_scaler.transform(Ytest.reshape(-1, 1)).flatten()

model = SVR(kernel='rbf')
model.fit(Xtrain, Ytrain)
print("train score:", model.score(Xtrain, Ytrain))
print("test score:", model.score(Xtest, Ytest))
