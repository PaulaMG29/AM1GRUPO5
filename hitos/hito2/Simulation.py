# ==============================================
# SIMULACIÓN
# ==============================================


import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from EsquemasTemporales import Euler, RK4, Crank_Nicolson, Inverse_Euler

# ---------- PARÁMETROS NUMÉRICOS DE LA SIMULACIÓN -----------

# Número de pasos temporales:
N = 100
# Número de variables del sistema --> U = (x1, x2):
Nv = 2
# Paso temporal:
Dt = 0.1

#-------------------------------------------------------------


# ---------------- DEFINICIÓN DE LA DINÁMICA -----------------

def F(U: array, t:float)-> array:
    return np.array([1 * U[1], -1 * U[0]])

U = np.zeros((N+1, Nv))
U[0, :] = np.array([1, 0])

#-------------------------------------------------------------


#----------------- BUCLE PRINCIPAL DE EULER ------------------

for n in range(0, N):
#    U[n+1, :] = Inverse_Euler(U[n, :], n*Dt, Dt, F)
    U[n+1, :] = Euler(U[n, :], n*Dt, Dt, F)
    # U[n+1, :] = RK4(U[n, :], n*Dt, Dt, F)
    # U[n+1, :] = Crank_Nicolson(U[n, :], n*Dt, Dt, F)
    
#-------------------------------------------------------------


# ----------------- REPRESENTACIÓN GRÁFICA -------------------

plt.plot(U[:, 0], U[:, 1])
plt.xlabel("x")
plt.ylabel(r'$\dot{x}$')
plt.axis('equal')
plt.grid()
plt.show()