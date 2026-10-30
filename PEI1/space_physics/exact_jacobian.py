from numpy import array

def exact_Jacobian(I1: float, I2: float, I3: float, omega: array) -> array:
    """
    Calcula la matriz jacobiana de las ecuaciones de Euler para un cuerpo rígido en rotación.
    
    Parámetros:
    I1, I2, I3: Momentos de inercia alrededor de los ejes principales.
    omega1, omega2, omega3: Componentes del vector de velocidad angular.
    
    Retorna:
    J: Matriz jacobiana de las ecuaciones de Euler.
    """
    J = array([[0, (I2 - I3) / I1 * omega[2], (I2 - I3) / I1 * omega[1]],
               [(I3 - I1) / I2 * omega[2], 0, (I3 - I1) / I2 * omega[0]],
               [(I1 - I2) / I3 * omega[1], (I1 - I2) / I3 * omega[0], 0]])
    
    return J
