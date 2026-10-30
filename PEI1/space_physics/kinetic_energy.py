from numpy import array

def T(I1: float, I2: float, I3: float, omega: array) -> array:
    """
    Calcula la energía cinética de rotación de un cuerpo rígido.
    
    Parámetros:
    I1, I2, I3: Momentos de inercia alrededor de los ejes principales.
    omega1, omega2, omega3: Componentes del vector de velocidad angular.
    
    Retorna:
    T: Energía cinética de rotación.
    """
    T = 0.5 * (I1 * omega[0]**2 + I2 * omega[1]**2 + I3 * omega[2]**2)
    
    return T
