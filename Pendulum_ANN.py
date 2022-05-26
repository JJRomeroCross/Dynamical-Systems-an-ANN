#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 21 22:59:23 2022

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

import seaborn as sns
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

m = 5000
i = 0
G = 9.8  # acceleration due to gravity, in m/s^2
L1 = 1.0  # length of pendulum 1 in m
L2 = 1.0  # length of pendulum 2 in m
L = L1 + L2  # maximal length of the combined pendulum
M1 = 1.0  # mass of pendulum 1 in kg
M2 = 1.0  # mass of pendulum 2 in kg
t_stop = 10  # how many seconds to simulate
history_len = 500  # how many trajectory points to display

""" Definimos las funciones que vamos a necesitar """
# Funcion para definir las ecuaciones del pendulo doble
def derivs(state, t):

    dydx = np.zeros_like(state)
    dydx[0] = state[1]

    delta = state[2] - state[0]
    den1 = (M1+M2) * L1 - M2 * L1 * cos(delta) * cos(delta)
    dydx[1] = ((M2 * L1 * state[1] * state[1] * sin(delta) * cos(delta)
                + M2 * G * sin(state[2]) * cos(delta)
                + M2 * L2 * state[3] * state[3] * sin(delta)
                - (M1+M2) * G * sin(state[0]))
               / den1)

    dydx[2] = state[3]

    den2 = (L2/L1) * den1
    dydx[3] = ((- M2 * L2 * state[3] * state[3] * sin(delta) * cos(delta)
                + (M1+M2) * G * sin(state[0]) * cos(delta)
                - (M1+M2) * L1 * state[1] * state[1] * sin(delta)
                - (M1+M2) * G * sin(state[2]))
               / den2)

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
    model.add(Dense(1000, input_dim = 4, activation = 'relu'))
    model.add(Dropout(taza_abandono))
    model.add(Dense(200, activation = 'relu'))
    model.add(Dropout(taza_abandono))
    model.add(Dense(1000, activation = 'relu'))
    model.add(Dropout(taza_abandono))
    model.add(Dense(200, activation = 'relu'))
    model.add(Dropout(taza_abandono))
    model.add(Dense(50, activation = 'relu'))
    model.add(Dropout(taza_abandono))
    model.add(Dense(4, activation = 'softmax'))
    
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

#model1 = Sequential([Dense(units = 4, input_shape = [datos.values.shape[1]])])

#model1.compile(optimizer='sgd', loss='mean_squared_error')

#model1.summary()
xTrain, yTrain, xTest, yTest = np.empty((0, 4)), np.empty((0, 4)), np.empty((0, 4)), np.empty((0, 4))
while(i<=m):
    
    # Generamos las condiciones inciales
    th1_train = 180.0*np.random.rand()
    w1_train = 180.0*np.random.rand()
    th2_train = 180.0*np.random.rand()
    w2_train = 180.0*np.random.rand()

    # initial state
    state = np.radians([th1_train, w1_train, th2_train, w2_train])

    # integrate your ODE using scipy.integrate.
    y = integrate.odeint(derivs, state, t)
    
    datos = pd.DataFrame({'x1(Ang1)': y[:, 0], 'x2(Vel1)': y[:, 1], 'x3(Ang2)': y[:, 2], 'x4(Vel2)': y[:, 3]})
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

th1 = -50.0
w1 = 0.0
th2 = -10.0
w2 = 0.0

# initial state
state = np.radians([th1, w1, th2, w2])

# integrate your ODE using scipy.integrate.
y = integrate.odeint(derivs, state, t)
datos_tiempo = pd.DataFrame({'x1(Ang1)': y[:, 0], 'x2(Vel1)': y[:, 1], 'x3(Ang2)': y[:, 2], 'x4(Vel2)': y[:, 3]})

# Volvemos a separar el dataset
r = 0.5
y_verdad, x_test, yTrain, yTest = separar_set(datos.values[0:n, :], datos.values[0:n, :], n, r)


# Sacamos las predicciones de la segunda mitad del dataset

#y_pred = 0.0
#y_verdad = state
#while(len(y_verdad)<len(t)):
#  i = len(y_verdad)
#  y_pred = model1(y_verdad[i-1:i])
#  y_verdad = np.concatenate((y_verdad, y_pred), axis = 0)
#  print(i)
#  i = i+1
#print(model1.predict(y_verdad[i-1:i]))
y_pred = model1.predict(y_verdad)
y_verdad = np.concatenate((y_verdad, y_pred), axis = 0) 

# Definimos las trayectorias del pendulo y creamos la animacion

x1 = L1*sin(y_verdad[:, 0])
y1 = -L1*cos(y_verdad[:, 0])

x2 = L2*sin(y_verdad[:, 2]) + x1
y2 = -L2*cos(y_verdad[:, 2]) + y1

fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(autoscale_on=False, xlim=(-L, L), ylim=(-L, 1.))
ax.set_aspect('equal')
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
trace, = ax.plot([], [], '.-', lw=1, ms=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
history_x, history_y = deque(maxlen=history_len), deque(maxlen=history_len)


def animate(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]

    if i == 0:
        history_x.clear()
        history_y.clear()

    history_x.appendleft(thisx[2])
    history_y.appendleft(thisy[2])

    line.set_data(thisx, thisy)
    trace.set_data(history_x, history_y)
    time_text.set_text(time_template % (i*dt))
    return line, trace, time_text


ani = animation.FuncAnimation(
    fig, animate, len(y), interval=dt*1000, blit=True)
ani.save('pendulo_segundo_intento.gif')
plt.show()
