# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 12:55:51 2019

@author: Ruchitesh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X,y)
regressor.predict(np.array([[6.5]]))

plt.scatter(X,y,color="red")
#plt.plot(X,regressor.predict(X),color="blue")

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=4)
X_poly=poly.fit_transform(X)
regressor_poly = LinearRegression()
regressor_poly.fit(X_poly,y)
plt.plot(X,regressor_poly.predict(X_poly),color="blue")