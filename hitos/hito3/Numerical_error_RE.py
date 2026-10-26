
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "hito2"))
from numpy import array, log, polyfit, zeros, inf
import matplotlib.pyplot as plt
from numpy.linalg import norm
from typing import Callable
from EsquemasTemporales import Euler, RK4, Crank_Nicolson, Inverse_Euler
from Fisica import Kepler_movement, Oscilator
from Cauchy_Problem import Cauchy_Problem
from Partition import partition

def estimate_scheme_order(Esquema: Callable, U0: array, F: Callable, t: array) -> float:
    number_meshes = 5
    N_values = zeros(number_meshes, dtype=int)
    for i in range(N_values.size):
        N_values[i] = (len(t) - 1) * 2**(i - 2)

    errors = []
    for N in N_values:
        # malla gruesa
        t1 = partition(t[0], t[-1], N)
        # malla fina
        t2 = partition(t[0], t[-1], 2 * N)
        U1, t1 = Cauchy_Problem(Esquema, U0, F, t1)
        U2, t2 = Cauchy_Problem(Esquema, U0, F, t2)

        U2 = U2[::2]

        E = U2 - U1
        errors.append(norm(E, inf))

    plt.plot(log(N_values), log(errors))
    plt.scatter(log(N_values), log(errors), color= 'black')
    plt.xlabel("ln(N)")
    plt.ylabel("ln(E)")
    plt.grid()
    plt.show() 

    # ajuste lineal
    coeffs = polyfit(log(N_values), log(errors), 1)
    slope = coeffs[0]
    order = -slope
    return order

def Richardson_Error(Esquema: Callable, U0: array, F: Callable, t: array)-> array:
    t1 = t
    t2 = partition(t[0], t[-1], 2 * (len(t) - 1))
    Dt = t[1] - t[0]
    # Solución con Dt:
    U1, t_1 = Cauchy_Problem(Esquema, U0, F, t1)
    # Solución con Dt /2:
    U2, t_2 = Cauchy_Problem(Esquema, U0, F, t2)

    # Comparamos cada 2 puntos de la solución con Dt/2
    U2 = U2[::2]

    Orden = estimate_scheme_order(Esquema, U0, F, t)
    Error = (U2 - U1) / (Dt**Orden - (Dt/2)**Orden) * Dt**Orden
    
    return Error, Orden
    

#U0 = array([1, 0, 0])
U0 = array([0, 1])
t = partition(0, 20, 1000)


Error, Orden = Richardson_Error(RK4, U0, Euler_Equations, t)
Error_max = max(norm(Error, axis=1))

print("Orden del esquema =", Orden) 
print("Error =", Error)   
print("Max error =",Error_max)

plt.plot(t, Error[:, 0])
plt.xlabel("t")
plt.ylabel(r'Richardson Error')
plt.grid()
plt.show()