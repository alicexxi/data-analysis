import numpy as np
from scipy.sparse import spdiags, eye

# Parameters
n = 8  # Number of points
L = 20  # Domain length (from -10 to 10, so length is 20)
dx = L / n  # Grid spacing

# Generate the matrix A for the Laplacian (second derivatives in x and y)
e = np.ones(n)
A_x = spdiags([e, -4*e, e], [-1, 0, 1], n, n) / dx**2
A_y = spdiags([e, -4*e, e], [-1, 0, 1], n, n) / dx**2

# Enforce periodic boundary conditions
A_x = A_x.tolil()
A_y = A_y.tolil()
A_x[0, -1] = 1 / dx**2
A_x[-1, 0] = 1 / dx**2
A_y[0, -1] = 1 / dx**2
A_y[-1, 0] = 1 / dx**2

# Convert to CSR format for efficient arithmetic operations
A_x = A_x.tocsr()
A_y = A_y.tocsr()

# Matrix A is the sum of A_x and A_y (discrete Laplacian operator)
A = A_x + A_y

# Generate the matrix B for the first derivative with respect to x
B = spdiags([-e, e], [-1, 1], n, n) / (2 * dx)
B = B.tolil()
B[0, -1] = -1 / (2 * dx)
B[-1, 0] = 1 / (2 * dx)
B = B.tocsr()

# Generate the matrix C for the first derivative with respect to y
C = spdiags([-e, e], [-1, 1], n, n) / (2 * dx)
C = C.tolil()
C[0, -1] = -1 / (2 * dx)
C[-1, 0] = 1 / (2 * dx)
C = C.tocsr()

# Print the matrices
print("Matrix A (Laplacian):\n", A.toarray())
print("Matrix B (d/dx):\n", B.toarray())
print("Matrix C (d/dy):\n", C.toarray())