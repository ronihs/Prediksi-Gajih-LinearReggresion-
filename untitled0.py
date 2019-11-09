# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 13:30:53 2019

@author: roni
"""

#import liblary
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#impotrt dataset

dataset = pd.read_csv('Daftar_gaji.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Gaji vs Pengalaman (Training set)")
plt.xlabel('tahun bekerja')
plt.ylabel('Gaji')
plt.show()


plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Gaji vs Pengalaman (Training set)")
plt.xlabel('tahun bekerja')
plt.ylabel('Gaji')
plt.show()

from sklearn.metrics import accuracy_score
print("Accuracy Score = " , accuracy_score(y_test, y_pred.around()))






