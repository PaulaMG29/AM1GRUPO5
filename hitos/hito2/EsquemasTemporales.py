from numpy import array
from typing import Callable
from numpy.linalg import norm
from our_Newton import our_Newton

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

#def Crank_Nicolson(U: array, t: float, Dt: float, F: Callable[[array,float],array]) -> array:
#    Y= U.copy()
#    while norm(Y - U - Dt/2*(F(U, t) + F(Y, t + Dt)))>1e-6:
#        R = Y - U - Dt/2 * (F(U, t) + F(Y, t + Dt))
#        Y = Y - R
#    return Y

def Inverse_Euler(U: array, t: float, Dt: float, F: Callable[[array,float],array]) -> array:
    Y = U.copy()
    while norm(Y - U - Dt * F(Y, t + Dt))>1e-6:
        R = Y - U - Dt * F(Y, t + Dt)
        Y = Y - R
    return Y