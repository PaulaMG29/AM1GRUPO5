# -*- coding: utf-8 -*-
# ==============================================================================
# OSCILADOR ARMÓNICO SIMPLE RESUELTO CON ESQUEMA NUMÉRICO RUNGE KUTTA DE ORDEN 4
# ==============================================================================

"""
Misma forma de proceder que en el Euler, solo cambia el esquema.
F(U) es la misma porque es el mismo problema del oscilador armónico.    
"""

from numpy import array, zeros
import matplotlib.pyplot as plt

N = 100
Nv = 2
Dt = 0.1

def F(U: array)-> array:
    return array([U[1], -U[0]])

U = zeros((N+1, Nv))
U[0, :] = array([1, 0])


"""
IDEA DE RUNGE-KUTTA DE ORDEN 4 (RK4):
En lugar de usar una sola pendiente (como en el Euler), calcula varias estimaciones 
dentro del intervalo temporal y luego hace un promedio ponderado.

Supongamos que queremos ir desde t_n hasta t_{n+1} (t_n + Dt)

Calcula k_1 (pendiente al inicio), k_2 (perdiente a la mitad del paso usando el valor
de k_1), k_3 (perdiente a la mitad del paso usando el valor de k_2), y k_4 (pendiente
al final usando k_3) 
                                                                            
Finalmente combina todas ellas.
"""
for n in range(0, N):
    k1 = F(U[n,:])
    k2 = F(U[n,:]+Dt*k1/2)
    k3 = F(U[n,:]+Dt*k2/2)
    k4 = F(U[n,:]+Dt*k3)
    
    U[n+1,:]=U[n,:]+Dt*(k1+2*k2+2*k3+k4)/6

plt.plot(U[:, 0], U[:, 1])
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.grid()
plt.show()
