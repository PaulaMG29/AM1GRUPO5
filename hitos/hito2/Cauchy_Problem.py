from numpy import array, zeros, linspace
import matplotlib.pyplot as plt
from typing import Callable
from .EsquemasTemporales import Euler, RK4, Crank_Nicolson, Inverse_Euler
from .Fisica import Kepler_movement

def Cauchy_Problem(Esquema: Callable, U0: array, F: Callable, Dt: float, Tf: float) -> array:
    Nv = len(U0)
    N = int(Tf/Dt)
    t = linspace(0, N*Dt, N+1)
    U = zeros((N+1, Nv))
    U[0, :] = U0
    for n in range(0, N):
        U[n+1, :] = Esquema(U[n, :], t[n], Dt, F)
    return U, t

U0 = array([1, 0, 0, 1])

U, t = Cauchy_Problem(Euler, U0, Kepler_movement, 0.0001, 10)

print (U)
print (t)

plt.plot(U[:, 0], U[:, 1])
plt.xlabel("x")
plt.ylabel(r'$\dot{x}$')
plt.axis('equal')
plt.grid()
plt.show()