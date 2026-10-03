from numpy import array
from math import sqrt

#====================================================================
# HARMONIC OSCILLATOR
#====================================================================

def Oscilator(U: array, t:float)-> array:
    return array([1 * U[1], -1 * U[0]])

#====================================================================
# KEPLER MOVEMENT
#====================================================================
"""
F = [dr/dt, -r/|r|^3]

F(U,t) = [dx/dt, dy/dt, -x /(x^2 + y^2)^[3/2], -y /(x^2 + y^2)^[3/2]] 
donde:
U = [x, y, dx/dt, dy/dt]
"""

def Kepler_movement(U: array, t:float)-> array:
    r = sqrt(U[0]**2 + U[1]**2)
    return array([U[2], U[3], -U[0]/r**3, -U[1]/r**3])