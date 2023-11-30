# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 12:16:43 2023

@author: usuario
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Cargar datos desde el archivo CSV
datos = pd.read_csv('DataBC.csv')

# Dividir los datos en características (X) y variable objetivo (y)
X = datos[['Puntuacion_Cliente']]
y = datos['Años_Entidad']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6 )

# Crear y entrenar el modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = modelo.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Visualizar los resultados
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.xlabel('Puntuacion Clente')
plt.ylabel('Edad')
plt.title('Modelo de Regresión Lineal')
plt.show()