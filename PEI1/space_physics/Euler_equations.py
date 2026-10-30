from numpy import array

# Las inercias se dan en el código de la GUI, son inputs del usuario.
# Aquí solo hay valores por defecto: la GUI los sobrescribe (space_physics.Euler_equations.I1 = ...)
# antes de integrar, para que Euler_Equations(omega, t) entre directamente en Cauchy_Problem.
I1, I2, I3 = 0.5, 1.0, 1.5

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
