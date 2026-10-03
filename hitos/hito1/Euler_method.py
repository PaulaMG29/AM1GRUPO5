# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 13:23:56 2026

@author: Paula
"""
# =============================================================
# OSCILADOR ARMÓNICO SIMPLE RESUELTO CON ESQUEMA NUMÉRICO EULER
# =============================================================
"""
Supongamos que tenemos una ecuación diferencial:
    dU/dt = F(U)
    donde:
    U es el vector de variables del sistema
    F(U) describe cómo cambian las variables del sistema

MÉTODO DE EULER:
    1) Calculas la pendiente actual
    2) Avanzas un pequeño paso usando esa pendiente
    3) Repites muchas veces
    
Fórmula del método de Euler:
    U_{n+1} = U_n + dt F(U_n)
    
    donde:
        U_n = vector de estado actual
        U_{n+1} = vector de estado siguiente
        dt = paso temporal
    
"""
print("Empiezo")
from numpy import array, zeros
#import matplotlib.pyplot as plt
print("Empiezo")
# ---------- PARÁMETROS NUMÉRICOS DE LA SIMULACIÓN -----------

# Número de pasos temporales:
N = 10
# Número de variables del sistema --> U = (x1, x2):
Nv = 2
# Paso temporal:
Dt = 0.1

#-------------------------------------------------------------


# ---------------- DEFINICIÓN DE LA DINÁMICA -----------------
"""
Trás linealizar el problema del oscilador armónico (ver apuntes de clase),
nos queda que F(U) = A U, donde A es una matriz.
    (0   1)
A = (-1  0)
"""
# Función F(U) = A U. Hace la multiplicación de la matrix A 
# (específica del oscilador armónico) por el vector U. 
def F(U: array)-> array:
    return array([1 * U[1], -1 * U[0]])

# Crea una matriz de ceros de N+1 filas y 2 columnas. Es decir, cada
# fila almacena el vector de estado en un instante:
U = zeros((N+1, Nv))

# Vector de estado con la condición inicial --> x_1 = 1, x_2 = 0:
# Por tanto, U_0 = (1, 0)
# Se almacena en la fila zero, y las 2 columnas de la matriz.
U[0, :] = array([1, 0])

#-------------------------------------------------------------


#----------------- BUCLE PRINCIPAL DE EULER ------------------
"""
Para cada paso temporal, calcula el vector de estado del paso siguiente
usando la fórmula del esquema de Euler.
"""
for n in range(0, N):
    U[n+1, :] = U[n, :] + Dt * F(U[n, :])
    
#-------------------------------------------------------------
print("Empiezo")
print(U)
print("Empiezo")
# ----------------- REPRESENTACIÓN GRÁFICA -------------------

# Diagrama de fases \dot{x} (que es x_2, la segunda componente del
# vector de estado), frente a x (x_1, primera componente), en cada
# instante temporal.

plt.plot(U[:, 0], U[:, 1])
plt.xlabel("x")
plt.ylabel(r'$\dot{x}$')
plt.axis('equal')
plt.grid()
#plt.show()
