import numpy as np
from time import time
from time import sleep
from traditional import *
from strassen import *

def crearMatriz(tam):
    return np.random.randint(0, 10, (tam, tam))

n = 7
A = crearMatriz(2**n)
B = crearMatriz(2**n)

if __name__ == "__main__":
    # Imprimir tamaño de matrices
    print(f"Multiplicando matrices de tamaño {A.shape[1]}x{B.shape[0]}\n")

    print("Starting traditional matrix multiplication...\n")
    inicio = time()
    result_traditional = traditional_matrix_multiplication(A, B)
    fin = time()
    #print(result_traditional)

    # Imprimir tiempo de ejecución
    print(f"Execution time: {fin - inicio:.8f} s\n")

    print("Waiting for next algorithm...")
    sleep(3)
    print("Resuming...\n")

    # Medir el tiempo de ejecución
    print("Starting Strassen multiplication\n")
    inicio = time()
    result_strassen = strassen_multiplication(A, B)
    fin = time()
    #print(result_strassen)

    # Imprimir tiempo de ejecución
    print(f"Execution time: {fin - inicio:.8f} s\n")

    print("Starting equality checking...\n")

    if np.array_equal(result_traditional, result_strassen):
        print("Matrices result_traditional and result_strassen are equal")
    else:
        print("Matrices result_traditional and result_strassen are not equal")

