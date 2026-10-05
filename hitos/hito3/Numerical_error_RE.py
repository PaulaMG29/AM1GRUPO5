
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "hito2"))
from numpy import array
from numpy.linalg import norm
from typing import Callable
from EsquemasTemporales import Euler, RK4, Crank_Nicolson, Inverse_Euler
from Fisica import Kepler_movement
from Cauchy_Problem import Cauchy_Problem

def Richardson_Error(Esquema: Callable, Orden: int, U0: array, F: Callable, Dt: float, Tf: float)-> array:
    # Solución con Dt:
    U1, t1 = Cauchy_Problem(Esquema, U0, F, Dt, Tf)
    # Solución con Dt /2:
    U2, t2 = Cauchy_Problem(Esquema, U0, F, Dt/2, Tf)
    
    # Comparamos cada 2 puntos de la solución con Dt/2
    U2 = U2[::2]
    
    Error = (U2 - U1) / (Dt**Orden - (Dt/2)**Orden) * Dt**Orden
    
    return Error
    

U0 = array([1, 0, 0, 1])


Error = Richardson_Error(Crank_Nicolson, 2, U0, Kepler_movement, 0.1, 10)
Error_max = max(norm(Error, axis=1))
    
print("Error =", Error)   
print("Max error =",Error_max)