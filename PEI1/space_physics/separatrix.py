from numpy import linspace, sqrt, cos, sin, pi, column_stack

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
