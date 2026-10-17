from numpy import array, zeros

A = array([[0, 1, -1],
           [0, -1, 2],
           [5, 1, 2]], dtype=float)
b = array([8, 3, 3], dtype=float)

def our_Gauss(A: array, b: array) -> array:
    """
    Solves the linear system Ax = b using Gaussian elimination with partial pivoting.
    Parameters:
    A : array
        Coefficient matrix.
    b : array
        Right-hand side vector.
    Returns:
    x : array
        Solution vector.
    """
    # n = número de ecuaciones (filas de la matriz A)
    n = len(b)
    # Eliminación
    for k in range(0,n-1):
        # Buscar pivote:
        pivot_row = k
        for i in range(k+1, n):
            if abs(A[i,k]) > abs(A[pivot_row,k]):
                pivot_row = i
        # Intercambiar filas si es necesario
        if pivot_row != k:
            A[[k, pivot_row]] = A[[pivot_row, k]]
            b[[k, pivot_row]] = b[[pivot_row, k]]
        # Verificar si el pivote es cero
        if abs(A[k,k]) < 1e-14:
            raise ValueError("Matriz singular.")
        # Eliminación
        for i in range(k+1, n):
            factor = A[i,k]/A[k,k]
            A[i,k:] = A[i,k:] - factor*A[k,k:]
            b[i] = b[i] - factor*b[k]
    # Sustitución hacia atrás
    x = zeros(n)
    for i in range(n-1, -1, -1):
        s = 0
        for j in range(i+1, n):
            s = s + A[i,j]*x[j]
        x[i] = (b[i] - s) / A[i,i]
    return x

A_diag,x = our_Gauss(A, b)
print("A_diag:", A_diag)
print("x:", x)