#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 11:54:39 2022

@author: juanjo
"""


""" Este es el mimso problema de la red neuronal y el péndulo, pero
esta vez vamos a entrenar la red neuronal generando por lo menos
m datasets con diferentes condiciones inciales que se van a generar
de forma aleatoria"""

from numpy import sin, cos
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
from collections import deque
from mpl_toolkits.mplot3d import Axes3D



import seaborn as sns
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

m = 10000
i = 0
t_stop = 10 # how many seconds to simulate

""" Definimos las funciones que vamos a necesitar """
# Funcion para definir las ecuaciones del atractor de Lorentz
def dLorenz(state, t):
    
    a, b, c = 10.0, 28.0, 8.0/3.0
    dydx = np.zeros_like(state)
    dydx[0] = a*(state[1]-state[0])
    dydx[1] = state[0]*(b-state[2]) - state[1]
    dydx[2] = state[0]*state[1] - c*state[2]
    
    return dydx

# Funcion para separar un dataset
def separar_set(x, y, n, r):
  xTrain, xTest, yTrain, yTest = [], [], [], []
  for i in range(len(x)):
    if (i<=r*float(n)):
      xTrain.append(x[i])
      yTrain.append(y[i])
    else:
      xTest.append(x[i])
      yTest.append(y[i])
  return np.array(xTrain), np.array(xTest), np.array(yTrain), np.array(yTest)

# Funcion para definir un modelo de red neuronal
def crear_modelo(taza_aprendizaje, taza_abandono):
    # Creamos el modelo
    model = Sequential()
    
    # Agregamos capas
    model.add(Dense(1000, input_dim = 3, activation = 'relu'))
    model.add(Dropout(taza_abandono))
    model.add(Dense(200, activation = 'relu'))
    model.add(Dropout(taza_abandono))
    model.add(Dense(1000, activation = 'relu'))
    model.add(Dropout(taza_abandono))
    model.add(Dense(200, activation = 'relu'))
    model.add(Dropout(taza_abandono))
    model.add(Dense(50, activation = 'relu'))
    model.add(Dropout(taza_abandono))
    model.add(Dense(3, activation = 'softmax'))
    
    # Compilamos el modelo
    adam = Adam(learning_rate = taza_aprendizaje)
    model.compile(optimizer = adam, metrics = ['accuracy'], loss = 'mean_squared_error')
    return model

# create a time array from 0..t_stop sampled at 0.02 second steps
dt = 0.02 # El salto temporal
t = np.arange(0, t_stop, dt)

# Empezamos resolviendo la ecuacion m veces para generar m datasets con
# los que se va a entrenar la red neuronal

taza_aprendizaje, taza_abandono = 0.001, 0.1
model1 = crear_modelo(taza_aprendizaje, taza_abandono)

xTrain, yTrain, xTest, yTest = np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3))
while(i<=m):
    
    #initial state
    state = np.array([30.0*np.random.rand(), 30.0*np.random.rand(), 30.0*np.random.rand()])
    # integrate your ODE using scipy.integrate.
    y = integrate.odeint(dLorenz, state, t)
    
    datos = pd.DataFrame({'x': y[:, 0], 'y': y[:, 1], 'z': y[:, 2]})
    r = 0.9
    n = len(y)
    xTrain_ind, xTest_ind, yTrain_ind, yTest_ind = separar_set(datos.values[0:n-1], datos.values[1:n], n, r)
    
    xTrain = np.concatenate((xTrain, xTrain_ind), axis = 0)
    xTest = np.concatenate((xTest, xTest_ind), axis = 0)
    yTrain = np.concatenate((yTrain, yTrain_ind), axis = 0)
    yTest = np.concatenate((yTest, yTest_ind), axis = 0)
    #model1.fit(xTrain, yTrain, epochs=100)
        
    i += 1
    
model1.fit(xTrain, yTrain, epochs=20, validation_data = (xTest, yTest))
model1.evaluate(xTest, yTest)    
# Ahora que ya hemos entrenado la red neuronal m veces, vamos a ponernos a 
# resolver las ecuaciones una última vez
    
""" Ahora que ya hemos entrenado el modelo, vamos a resolver de nuevo
las ecuaciones, pero con otras condiciones iniciales"""

# initial state
state0 = np.array([1.0, 1.0, 1.0])

# integrate your ODE using scipy.integrate.
y = integrate.odeint(dLorenz, state, t)

datos_tiempo = pd.DataFrame({'Tiempo': t, 'x': y[:, 0], 'y': y[:, 1], 'z': y[:, 2]})

datos = datos_tiempo.drop(['Tiempo'], axis = 1)
# Volvemos a separar el dataset
r = 0.5
y_verdad, x_test, yTrain, yTest = separar_set(datos.values[0:n, :], datos.values[0:n, :], n, r)


# Sacamos las predicciones de la segunda mitad del dataset

y_pred = 0.0
y_verdad = datos.values[0:2]
#y_verdad = state
#y_verdad = np.append(y_verdad, datos.values[1:2], axis = 0)
while(len(y_verdad)<len(t)):
  i = len(y_verdad)
  y_pred = model1(y_verdad[i-1:i])
  y_verdad = np.append(y_verdad, y_pred, axis = 0)
#  print(i)
#  i = i+1
#print(model1.predict(y_verdad[i-1:i]))
#y_pred = model1.predict(y_verdad[:, :])
#y_verdad = np.append((y_verdad, y_pred), axis = 0) 

# Creamos la animacion

dataSet = np.array([y_verdad[:, 0], y_verdad[:, 1], y_verdad[:, 2]])
fig = plt.figure()
ax = Axes3D(fig)
redDots = plt.plot(dataSet[0], dataSet[1], dataSet[2], lw = 2, c = 'r', marker = 'o')[0]
line = plt.plot(dataSet[0], dataSet[1], dataSet[2], lw = 2, c = 'g')[0]

time_template = 'time = %.1fs'
time_text = ax.text(5.0, 5.0, 5.0, '', transform=ax.transAxes)
#history_x, history_y, history_z = deque(maxlen=history_len), deque(maxlen=history_len)


def animate(i, dataSet, line, redDots):
    line.set_data(dataSet[0:2, :i])
    line.set_3d_properties(dataSet[2, :i])
    redDots.set_data(dataSet[0:2, :i])
    redDots.set_3d_properties(dataSet[2, :i])

    time_text.set_text(time_template % (i*dt))
    return line, time_text

ani = animation.FuncAnimation(
    fig, animate, len(y), fargs = (dataSet, line, redDots),interval=dt*1000, blit=False)
ani.save('Lorenzo_ANN.gif')
plt.show()