# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 18:01:18 2021

@author: Ayush Patel
"""

import pandas as pd  
import numpy as np    
import matplotlib.pyplot as plt    
from sklearn.model_selection import train_test_split   
from sklearn.linear_model import LinearRegression 
 
data_load = pd.read_csv(r"D:\New folder\student.csv")  
print("Successfully imported data into console" ) 
data=data_load.head(6)
print(data)
data_load.plot(x='Hours', y='Scores', color='green',style='o')    
plt.title('Hours vs Percentage')    
plt.xlabel('The Hours Studied')    
plt.ylabel('The Percentage Score')    
plt.show()

X = data_load.iloc[:, :-1].values    
y = data_load.iloc[:, 1].values 

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0)

reg = LinearRegression()    
reg.fit(X_train, y_train)  
line = reg.coef_*X+reg.intercept_  
plt.scatter(X, y)  
plt.plot(X, line);  
plt.show() 

hours = [[8]]  
own_pred = reg.predict(hours)  
print("Number of hours = {}".format(hours))  
print("Prediction Score = {}".format(own_pred[0])) 