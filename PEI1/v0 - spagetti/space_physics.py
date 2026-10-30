from numpy import array, linspace, zeros, sqrt, cos, sin, pi, abs, column_stack, concatenate
from numpy.linalg import norm, eigvals, svd

# Las inercias se dan en el código de gui_reactive.py, son inputs del usuario.
# Aquí solo hay valores por defecto: la GUI los sobrescribe (space_physics.I1 = ...)
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


def Attitude_Equations(U: array, t: float) -> array:
    """
    Dinámica (Euler) y cinemática de actitud del sólido rígido.

    U = [omega1, omega2, omega3, R11, R12, R13, R21, ..., R33], con R la matriz de
    actitud (ejes cuerpo -> inerciales) por filas. Usa las inercias globales I1, I2, I3.

    Retorna:
    dU/dt = [d_omega/dt, dR/dt], con dR/dt = R [omega]x
    """
    omega = U[0:3]
    R = U[3:12].reshape(3, 3)
    omega_skew = array([[0, -omega[2], omega[1]],
                        [omega[2], 0, -omega[0]],
                        [-omega[1], omega[0], 0]])
    return concatenate([Euler_Equations(omega, t), (R @ omega_skew).reshape(9)])

def nearest_rotation(R: array) -> array:
    """
    Rotación más cercana a R (descomposición polar vía SVD). Los esquemas no conservan
    exactamente la ortogonalidad de R; esta proyección permite dibujar la actitud.
    """
    W, _, Vt = svd(R)
    return W @ Vt

def separatrix(I1: float, I2: float, I3: float, L0: float, N: int = 361) -> list:
    """
    Separatriz crítica sobre el elipsoide de momento |L| = L0 (nivel 2T = L0²/I2).

    Es la intersección del elipsoide de momento con los planos
    sqrt(I1 (I2 - I1)) omega1 = ± sqrt(I3 (I3 - I2)) omega3: dos elipses que pasan por ±omega*(2).

    Retorna:
    Lista con las dos curvas (arrays N x 3), vacía si L0 = 0 o no se cumple I1 < I2 < I3.
    """
    if L0 <= 0 or not (I1 < I2 < I3):
        return []
    k = sqrt(I1 * (I2 - I1) / (I3 * (I3 - I2)))
    theta = linspace(0., 2 * pi, N)
    omega1 = L0 * cos(theta) / sqrt(I1**2 + (I3 * k)**2)
    omega2 = L0 * sin(theta) / I2
    return [column_stack([omega1, omega2, sign * k * omega1]) for sign in (-1., 1.)]

def equilibrium_spectrum(I1: float, I2: float, I3: float, Omega: float, k: int) -> tuple:
    """
    Linealización en el equilibrio omega*(k) = Omega e_k (k = 0, 1, 2).

    Retorna:
    (autovalores de exact_Jacobian, True si hay algún autovalor con parte real > 0 -> inestable)
    """
    omega = zeros(3)
    omega[k] = Omega
    eigenvalues = eigvals(exact_Jacobian(I1, I2, I3, omega))
    return eigenvalues, bool(max(eigenvalues.real) > 1e-9)

def invariants_drift(I1: float, I2: float, I3: float, omega: array) -> tuple:
    """
    Verificación de los invariantes a lo largo de una solución omega (N x 3).

    Retorna:
    (|2T(t) - 2T0| / 2T0, |L²(t) - L0²| / L0²)
    """
    energy = array([2 * T(I1, I2, I3, w) for w in omega])
    momentum = array([L(I1, I2, I3, w)**2 for w in omega])
    return abs(energy - energy[0]) / abs(energy[0]), abs(momentum - momentum[0]) / abs(momentum[0])
