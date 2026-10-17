from numpy import array, zeros
from typing import Callable
from numpy.linalg import LinAlgError, solve, norm

#================================================================================================
# LINEAR SOLVERS
#================================================================================================


def partition(a, b, N):
    """
    a = punto inicial
    b = punto final
    N = numero de segmentos
    """
    return array([a + i*(b-a)/N for i in range(0,N+1)])

#================================================================================================
# NON-LINEAR SOLVERS
#================================================================================================

def our_jacobian(F: Callable[[array,float],array], U: array, t: float) -> array:
    """
    F: function
    U: current state vector
    """
    n = len(U)
    J = zeros((n, n))
    h = 1e-8  # small perturbation for numerical differentiation

    for i in range(n):
        U_perturbed_forward = U.copy()
        U_perturbed_forward[i] += h
        U_perturbed_backward = U.copy()
        U_perturbed_backward[i] -= h
        J[:,i] = (F(U_perturbed_forward,t) - F(U_perturbed_backward, t)) / (2 * h)

    return J

def our_Newton(F: Callable[[array,float],array], U0: array, t: float, tol: float, max_iter: int) -> array:
    """
    F: function
    U0: initial guess for the root
    t: time parameter
    tol: tolerance for convergence
    max_iter: maximum number of iterations
    """
    U = U0.copy()
    
    for iteration in range(max_iter):
        J = our_jacobian(F, U, t)
        F_val = F(U, t)
        
        # Solve J * delta_U = -F_val for delta_U
        try:
            delta_U = -solve(J, F_val)
        except LinAlgError:
            raise ValueError("Jacobian is singular or ill-conditioned.")
        
        U += delta_U
        
        if norm(delta_U) < tol:
            return U
    
    raise ValueError("Newton's method did not converge within the maximum number of iterations.")

#================================================================================================
# NUMERICAL SCHEMES
#================================================================================================

def Euler(U: array, t: float, Dt: float, F: Callable[[array,float],array]) -> array:
    return U + Dt * F(U, t)

def RK4(U: array, t: float, Dt: float, F: Callable[[array,float],array]) -> array:
    k1 = F(U, t)
    k2 = F(U + Dt*k1/2, t+Dt/2)
    k3 = F(U + Dt*k2/2, t+Dt/2)
    k4 = F(U + Dt*k3, t+Dt/2)
    return U + Dt*(k1+2*k2+2*k3+k4)/6

def Crank_Nicolson(U: array, t: float, Dt: float, F: Callable[[array,float],array]) -> array:
    """
    G = function
    U = seed
    """
    def G(Y, t):
        return Y + A - Dt/2*(F(Y, t + Dt))
    A = - U - Dt/2 * F(U, t)
    return our_Newton(G, U, 0, 1e-6, 100)

#================================================================================================
# CAUCHY PROBLEM
#================================================================================================

def Cauchy_Problem(Esquema: Callable, U0: array, F: Callable, t: array) -> array:
    """
    t   : time domain (vector con todos los instantes temporales)
    """
    Nv = len(U0)
    N = len(t) - 1
    U = zeros((N+1, Nv))
    U[0, :] = U0
    for n in range(0, N):
        U[n+1, :] = Esquema(U[n, :], t[n], t[n+1] - t[n], F)
    return U, t

#================================================================================================
# ERROR ESTIMATION
#================================================================================================

def Richardson_Error(Esquema: Callable, Orden: int, U0: array, F: Callable, t: array)-> array:
    t1 = t
    t2 = partition(t[0], t[-1], 2 * (len(t) - 1))
    Dt = t[1] - t[0]
    # Solución con Dt:
    U1, t_1 = Cauchy_Problem(Esquema, U0, F, t1)
    # Solución con Dt /2:
    U2, t_2 = Cauchy_Problem(Esquema, U0, F, t2)
    
    # Comparamos cada 2 puntos de la solución con Dt/2
    U2 = U2[::2]
    
    Error = (U2 - U1) / (Dt**Orden - (Dt/2)**Orden) * Dt**Orden
    # Matriz de errores de Richardson de cada componente de U para cada instante temporal
    return Error

#================================================================================================
# NUMERICAL STABILITY REGION
#================================================================================================
