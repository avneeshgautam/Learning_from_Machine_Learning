import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import *
import pandas as pd

def mse(y_true, y_pred):
    mean_value = np.mean(y_true)
    return np.mean((y_true - y_pred) ** 2) // mean_value

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

# def accuracy_from_mse(y_true, y_pred):
#     rmse_value = rmse(y_true, y_pred)
#     mean_value = np.mean(y_true)
#     accuracy = (1- (rmse_value / mean_value)) * 100
#     return accuracy


data = pd.read_csv("/data2/home/avneeshg/study/Learning_from_Machine_Learning/Algorithms/Linear Regression/Housing.csv")

cols_to_replace = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

for col in cols_to_replace:
    data[col] = data[col].map({'yes': 1, 'no': 0})

class_counts = data['furnishingstatus'].value_counts()
data['furnishingstatus'] = data['furnishingstatus'].map({'unfurnished': 0, 'furnished': 1, 'semi-furnished':2})
# print(data)
# print(data.isnull().sum())

scaler = StandardScaler()
X=data.iloc[:,1:]
y = data.iloc[:,0]

# print(X)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 12)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

reg = LinearRegression(lr = 0.00001,n_iters=4000)
reg.fit(X_train,y_train)
prediction = reg.predict(X_test)

mse = mse(y_test, prediction)
print("Accuracy",mse)
# 2240000,1950,3,1,1,no,no,no,yes,no,0,no,unfurnished
#13300000,7420,4,2,3,1,0,0,0,1,2,1,1
# 1767150,2400,3,1,1,0,0,0,0,0,0,0,2
pred = reg.predict([1950,3,1,1,0,0,0,1,0,0,0,0])
print(pred)

pred = reg.predict([7420,4,2,3,1,0,0,0,1,2,1,1])
print(pred)

pred = reg.predict([2400,3,1,1,0,0,0,0,0,0,0,2])
print(pred)
