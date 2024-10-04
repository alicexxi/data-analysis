import numpy as np

x = np.array([-1.6]) 
A3 = [0, 0]

for j in range(100):
    x_new = x[j] - (x[j] * np.sin(3 * x[j]) - np.exp(x[j]))/(np.sin(3 * x[j]) + 3 * x[j] * np.cos(3*x[j]) - np.exp(x[j]))
    fc = x[j] * np.sin(3 * x[j]) - np.exp(x[j])
    x = np.append(x, x_new)
    A3[0] = j + 1
    if (abs(fc) < 1e-6):
        break

xr = -0.4; xl = -0.7
A2 = []
for j in range(100):
    xc = (xr + xl)/2
    fc = xc * np.sin(3 * xc) - np.exp(xc)
    if ( fc > 0 ):
        xl = xc
    else:
        xr = xc
    A2.append(xc)
    if ( abs(fc) < 1e-6 ):
        A3[1]= j + 1 
        break



A1 = x
A2 = A2

print("A1", A1)
print("A2", A2)
print("A3", A3)

A = np.array([[1, 2], [-1, 1]])
B = np.array([[2, 0], [0, 2]])
C = np.array([[2, 0, -3], [0, 0, -1]])
D = np.array([[1, 2], [2, 3], [-1, 0]])

x = np.array([1, 0]) 
y = np.array([0, 1])
z = np.array([1, 2, -1])

A4 = A + B

A5 = 3 * x - 4 * y

A6 = np.dot(A, x)

A7 = np.dot(B, x - y)

A8 = np.dot(D, x)

A9 = np.dot(D, y) + z

A10 = np.dot(A, B)

A11 = np.dot(B, C)

A12 = np.dot(C, D)

print("A4: A + B =\n", A4)
print("\nA5: 3x - 4y =\n", A5)
print("\nA6: A * x =\n", A6)
print("\nA7: B * (x - y) =\n", A7)
print("\nA8: D * x =\n", A8)
print("\nA9: D * y + z =\n", A9)
print("\nA10: A * B =\n", A10)
print("\nA11: B * C =\n", A11)
print("\nA12: C * D =\n", A12)
