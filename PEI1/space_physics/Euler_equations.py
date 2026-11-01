from numpy import array

# Las inercias se dan en el código de la GUI, son inputs del usuario.
# Aquí solo hay valores por defecto: la GUI los sobrescribe (space_physics.Euler_equations.I1 = ...)
# antes de integrar, para que Euler_Equations(omega, t) entre directamente en Cauchy_Problem.
# Quien necesite otras inercias sin tocar estas globales las pasa en `inertia`.
I1, I2, I3 = 0.5, 1.0, 1.5

#def Euler_Equations(I1: float, I2: float, I3: float, omega: array) -> array:
def Euler_Equations(omega: array, t:array, inertia: tuple = None) -> array:
    """
    Ecuaciones de Euler para un cuerpo rígido en rotación.
    
    Parámetros:
    I1, I2, I3: Momentos de inercia alrededor de los ejes principales.
    omega1, omega2, omega3: Componentes del vector de velocidad angular.
    inertia: (I1, I2, I3) opcional; si es None se usan las inercias globales del módulo.
    
    Retorna:
    d_omega1_dt, d_omega2_dt, d_omega3_dt: Derivadas temporales de las componentes de la velocidad angular.
    """
    i1, i2, i3 = (I1, I2, I3) if inertia is None else inertia
    d_omega1_dt = ((i2 - i3) / i1) * omega[1] * omega[2]
    d_omega2_dt = ((i3 - i1) / i2) * omega[2] * omega[0]
    d_omega3_dt = ((i1 - i2) / i3) * omega[0] * omega[1]
    
    return array([d_omega1_dt, d_omega2_dt, d_omega3_dt])
