import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression
from sklearn.datasets import *
import pandas as pd

def mse(y_test, prediction):
    return np.mean((y_test-prediction)**2)

data = pd.read_csv("/Users/avneeshgautam29/Placement_learning/Machine Learning/Learning_from_Machine_Learning/Algorithms/Linear Regression/Housing.csv")

cols_to_replace = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

for col in cols_to_replace:
    data[col] = data[col].map({'yes': 1, 'no': 0})

class_counts = data['furnishingstatus'].value_counts()
data['furnishingstatus'] = data['furnishingstatus'].map({'unfurnished': 0, 'furnished': 1, 'semi-furnished':2})
# print(data)
# print(data.isnull().sum())


X=data.iloc[:,1:]
y = data.iloc[:,0]

# print(X)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 12)
reg = LinearRegression(lr = 0.001)
reg.fit(X_train,y_train)
prediction = reg.predict(X_test)

mse = mse(y_test, prediction)
print(mse)
# 2240000,1950,3,1,1,no,no,no,yes,no,0,no,unfurnished
pred = reg.predict([1950,3,1,1,0,0,0,1,0,0,0,0])
print(pred)
