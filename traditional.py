import numpy as np

def traditional_matrix_multiplication(matrix1, matrix2):
  """Multiplies two matrices.

  Args:
    matrix1: The first matrix.
    matrix2: The second matrix.

  Returns:
    The product of the two matrices.

  Raises:
    ValueError: If the matrices have incompatible dimensions.
  """

  if matrix1.shape[1] != matrix2.shape[0]:
    raise ValueError("Matrices cannot be multiplied because of dimensions criteria.")

  result = np.zeros((matrix1.shape[0], matrix2.shape[1]))
  for i in range(matrix1.shape[0]):
    for j in range(matrix2.shape[1]):
      for k in range(matrix1.shape[1]):
        result[i, j] += matrix1[i, k] * matrix2[k, j]

  return result

