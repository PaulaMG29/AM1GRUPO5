from numpy import array, concatenate
from numpy.linalg import svd

from .Euler_equations import Euler_Equations

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
