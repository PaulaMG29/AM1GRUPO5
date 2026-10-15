from numpy import array

def partition(a, b, N):
    """
    a = punto inicial
    b = punto final
    N = numero de segmentos
    """
    return array([a + i*(b-a)/N for i in range(0,N+1)])