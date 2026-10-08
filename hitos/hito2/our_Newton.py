

from typing import Callable
from numpy import array, zeros
from numpy.linalg import LinAlgError, solve, norm

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
