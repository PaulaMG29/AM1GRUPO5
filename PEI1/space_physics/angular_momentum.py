from numpy import array
from numpy.linalg import norm

def L(I1: float, I2: float, I3: float, omega: array) -> array:
    """
    Calcula el momento angular de un cuerpo rígido en rotación.
    
    Parámetros:
    I1, I2, I3: Momentos de inercia alrededor de los ejes principales.
    omega1, omega2, omega3: Componentes del vector de velocidad angular.
    
    Retorna:
    L: Vector de momento angular.
    """
    L = array([I1 * omega[0], I2 * omega[1], I3 * omega[2]])
    
    return norm(L)
