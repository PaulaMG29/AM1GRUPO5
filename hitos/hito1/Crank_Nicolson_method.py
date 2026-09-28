# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""
# ======================================================
# OSCILADOR ARMÓNICO SIMPLE RESUELTO POR CRANK-NICOLSON
# ======================================================

"""
Misma forma de proceder que en el Euler, solo cambia el esquema.
F(U) es la misma porque es el mismo problema del oscilador armónico.    
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

N = 100
Nv = 2
Dt = 0.1

def F(U):
    return np.array([U[1], -U[0]])

U = np.zeros((N+1, Nv))
U[0, :] = np.array([1, 0])

"""
IDEA DE CRANK-NICOLSON:
En lugar de usar una sola pendiente al inicio (como en el Euler), 
calcula la media de las pendientes al principio y al final.

Es un método implícito --> hay que resolver iterativamente.
"""

for n in range(0, N):
    # Inicio del paso (aproximación de U_{n+1})
    # Inicialmente se toma U_{n+1} = U_n
    Y= U[n, :]
    
    # Queremos resolver: U_{n+1} = U_n + Dt/2 * (F(U_n) + F(U_{n+1}))
    # Llamando Y = U_{n+1}, nos queda:
    # Y - U_n - Dt/2 * (F(U_n) + F(Y)) = R
    # Donde R es el residuo que queremos que sea muy próximo a cero.
    while norm(Y - U[n,:]-Dt/2*(F(U[n,:])+F(Y)))>1e-6:
        R = Y - U[n,:]-Dt/2*(F(U[n,:])+F(Y))
        Y=Y-R
        # Y_new = Y_old - R. Para este problema concreto sí funciona porque
        # F(U) es lineal. En problemas más complejos hay que usar 
        # Newton-Raphson.
    U[n+1, :] = Y

plt.plot(U[:, 0], U[:, 1])
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.grid()
plt.show()
