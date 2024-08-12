import numpy as np
import pandas as pd
from KNN import KNN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("Social_Network_Ads.csv")
X = data.iloc[:,2:4].values
y = data.iloc[:,-1].values
# print("Shape of X : ",X.shape)
# print("Shape of y : ",y.shape)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)
# print("Shape of X_train : ",X_train.shape)
# print("Shape of y_train : ",y_train.shape)
# print("Shape of X_test : ",X_test.shape)
# print("Shape of y_test : ",y_test.shape)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

knn = KNN(k=5)
knn.fit(X_train,y_train)

age = int(input("Enter age: "))
salary = int(input("Enter sallary: "))
result = knn.predict([[age,salary]])

print(result)