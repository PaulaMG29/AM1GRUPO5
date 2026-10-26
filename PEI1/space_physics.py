from numpy import array, linspace
from numpy.linalg import norm

# Las inercias se dan en el código de gui_reactive.py, son inputs del usuario

#def Euler_Equations(I1: float, I2: float, I3: float, omega: array) -> array:
def Euler_Equations(omega: array, t:array) -> array:
    """
    Ecuaciones de Euler para un cuerpo rígido en rotación.
    
    Parámetros:
    I1, I2, I3: Momentos de inercia alrededor de los ejes principales.
    omega1, omega2, omega3: Componentes del vector de velocidad angular.
    
    Retorna:
    d_omega1_dt, d_omega2_dt, d_omega3_dt: Derivadas temporales de las componentes de la velocidad angular.
    """
    d_omega1_dt = ((I2 - I3) / I1) * omega[1] * omega[2]
    d_omega2_dt = ((I3 - I1) / I2) * omega[2] * omega[0]
    d_omega3_dt = ((I1 - I2) / I3) * omega[0] * omega[1]
    
    return array([d_omega1_dt, d_omega2_dt, d_omega3_dt])

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