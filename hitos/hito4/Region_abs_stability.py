import sys
from pathlib import Path
from typing import Callable
sys.path.append(str(Path(__file__).parent.parent / "hito2"))
from EsquemasTemporales import Euler, RK4, Crank_Nicolson
from Partition import partition
from numpy import array, linspace, meshgrid, abs, zeros_like
import matplotlib.pyplot as plt

def stability_region(scheme:Callable, xlim:tuple, ylim:tuple, N:int):
    x = linspace(xlim[0], xlim[1], N)
    y = linspace(ylim[0], ylim[1], N)
    X, Y = meshgrid(x, y)
    Z = X + 1j*Y

    R = zeros_like(Z, dtype=complex)

    for i in range(N):
        for j in range(N):
            z = Z[i,j]
            # Ecuación test, solo para sacar la región de estabilidad, no es parte del esquema
            # omega = Dt * lamda
            # Como Dt = 1, entonces omega = lamda, y la ecuación test es U'= F = z*U
            def F(U, t):
                return z*U
            # Tomando U0 = 1, el valor obtenido tras un paso (U1) es 
            # directamente el factor de amplificación R(z).
            U0 = array([1+0j])
            # U1 = scheme(U, t, Dt, F)
            U1 = scheme(U0, 0.0, 1.0, F)
            R[i,j] = U1[0]

    # La región de estabilidad viene dada por los puntos que 
    # cumplen |R(z)| ≤ 1.
    stable = abs(R) <= 1

    plt.contourf(X, Y, stable, levels=[0.5,1], colors=["lightblue"])
    plt.contour(X, Y, abs(R), levels=[1], colors="k")
    plt.axhline(0, linestyle= '--', color= "grey")
    plt.axvline(0, linestyle= '--', color= "grey")
    plt.xlabel(r"$Re(z)$")
    plt.ylabel(r"$Im(z)$")
    plt.axis("equal")
    plt.title(f"Stability Region of {scheme.__name__}")
    plt.show()

# Poner N = numero par para obtener la region de estabilidad de Crank-Nicolson
stability_region(RK4, xlim=(-4,4), ylim=(-4,4), N=1000)