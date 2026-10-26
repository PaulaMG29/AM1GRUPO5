import sys
from pathlib import Path
from typing import Callable
sys.path.append(str(Path(__file__).parent.parent / "hito2"))
from EsquemasTemporales import Euler, RK4, Crank_Nicolson
from Partition import partition
from numpy import array, linspace, meshgrid, abs, zeros_like
from numpy.linalg import eigvals
import matplotlib.pyplot as plt

def create_mesh(xlim:tuple, ylim:tuple, N:int):
    x = linspace(xlim[0], xlim[1], N)
    y = linspace(ylim[0], ylim[1], N)
    X, Y = meshgrid(x, y)
    return X, Y

def stability_region(scheme:Callable, X: array, Y: array):
    N = X.shape[0]
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
    return stable, R

def plot_stability_region(xlim, ylim, N, scheme:Callable, A: array, Dt: int):
    X, Y = create_mesh(xlim, ylim, N)
    stable, R = stability_region(scheme, X, Y)
    eigenvalues = eigvals(A)
    plt.contourf(X, Y, stable, levels=[0.5,1], colors=["lightblue"])
    plt.contour(X, Y, abs(R), levels=[1], colors="k")
    plt.axhline(0, linestyle= '--', color= "grey")
    plt.axvline(0, linestyle= '--', color= "grey")
    plt.scatter(eigenvalues.real*Dt, eigenvalues.imag*Dt, color="purple", label="Eigenvalues of A")
    plt.xlabel(r"$Re(z)$")
    plt.ylabel(r"$Im(z)$")
    plt.axis("equal")
    plt.title(f"Stability Region of {scheme.__name__}")
    plt.show()  

A = array([[0, 1], [-1, 0]])
# Poner N = numero par para obtener la region de estabilidad de Crank_Nicolson
plot_stability_region(xlim=(-4,4), ylim=(-4,4), N=50, scheme= Crank_Nicolson, A = array([[0, 1], [-1, 0]]), Dt=2.0)