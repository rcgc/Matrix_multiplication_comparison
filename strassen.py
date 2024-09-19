import numpy as np
from multiprocessing import Pool, cpu_count

# Divide la matriz en 4 subcuadrantes
def dividirCuadrante(matriz):
    row, col = matriz.shape
    mid_row, mid_col = row // 2, col // 2
    return matriz[:mid_row, :mid_col], matriz[:mid_row, mid_col:], matriz[mid_row:, :mid_col], matriz[mid_row:, mid_col:]

# Función para multiplicar submatrices en paralelo
def multiplicacion_paralelo(args):
    A, B = args
    return A.dot(B)

# Multiplicación de matrices usando divide y vencerás
def strassen_multiplication(A, B):
    # Caso base: Si la matriz es 1x1
    if len(A) == 1:
        return A * B

    # Dividir ambas matrices en submatrices
    a, b, c, d = dividirCuadrante(A)
    e, f, g, h = dividirCuadrante(B)

    # Crear un pool de procesos y realizar las multiplicaciones en paralelo
    # cpu_count()
    with Pool(1) as pool:
        resultados = pool.map(multiplicacion_paralelo, [(a, e), (b, g), (a, f), (b, h), 
                                                        (c, e), (d, g), (c, f), (d, h)])

    # Obtener los resultados de las submatrices
    p1, p2, p3, p4, p5, p6, p7, p8 = resultados

    # Combinar los resultados para formar la matriz resultante
    top = np.hstack((p1 + p2, p3 + p4))
    bottom = np.hstack((p5 + p6, p7 + p8))
    return np.vstack((top, bottom))

