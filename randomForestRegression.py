# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 20:58:04 2020

@author: suyog
"""

#Random Forest Regression

#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Import Data Set
data = pd.read_csv('Position_Salaries.csv')
X = data.iloc[:, 1:2].values
y = data.iloc[:, 2].values

"""
Splitting is not necessary because we have little data ( 10 rows )

#Splitting data set to train set and test set
from sklearn.cross_validation import train_test_split
X_train,y_train,X_test,y_test = train_test_split(X,y,test_size=0.3,random_state=0)"""

#Fitting Random Forest Regression to dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100,random_state=0)
regressor.fit(X,y)

#Predicting new result
y_pred = regressor.predict(X)
y_pred_6half = regressor.predict([[6.5]])

#Visualizing the Random Forest Regression Results 
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('Random Forest Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()