# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 21:07:30 2023

@author: usuario
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 20:54:14 2023

@author: usuario
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

datos=pd.read_csv('databc.csv')
df=pd.DataFrame(datos)
x=df['Años_Entidad']
y=df['Puntuacion_Cliente']
print("Datos Originales")
print(df)

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.6)
print("X_train")
print(X_train)
print("y_train")
print(y_train)

X_train=X_train.values.reshape([X_train.values.shape[0],1])
X_test=X_test.values.reshape([X_test.values.shape[0],1])

regr=linear_model.LinearRegression()
regr.fit(X_train,y_train)
y_pred=regr.predict(X_test)



print("Error: ", mean_squared_error(y_test, y_pred))
print("El valor de r2: ", r2_score(y_test,y_pred))