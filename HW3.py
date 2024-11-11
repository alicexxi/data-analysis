from scipy.integrate import solve_ivp
import numpy as np
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt


# part a
def hw3_rhs_a(t, y, E):
    return np.array([y[1], (t ** 2 - E) * y[0]])


tol = 1e-4
L = 4
x = np.arange(-L, L + 0.1, 0.1)
n = len(x)

E = 0.1
E_sola = np.zeros(5)
y_sola = np.zeros((n, 5))

for jmodes in range(5):
    dE = 0.2
    for j in range(1000):
        y0 = [1, np.sqrt(L ** 2 - E)]
        sol = solve_ivp(lambda x, y: hw3_rhs_a(x, y, E), [x[0], x[-1]], y0, t_eval=x)
        ys = sol.y.T
        bc = ys[-1, 1] + np.sqrt(L ** 2 - E) * ys[-1, 0]

        if abs(bc) < tol:
            break

        if (-1) ** jmodes * bc > 0:
            E += dE
        else:
            E -= dE
            dE /= 2

    E_sola[jmodes] = E
    norm = np.sqrt(np.trapezoid(ys[:, 0] ** 2, x=x))
    y_sola[:, jmodes] = np.abs(ys[:, 0] / norm)
    E += 0.2

A1 = y_sola
A2 = E_sola
print('A1 = \n', y_sola)
print('A2 = \n', E_sola)

# Part b
L = 4
dx = 0.1
x = np.arange(-L, L + dx, dx)
N = len(x) - 2

M = np.zeros((N, N))
for i in range(N):
    M[i, i] = -2 - (x[i + 1] ** 2) * (dx ** 2)

for i in range(N - 1):
    M[i, i + 1] = 1
    M[i + 1, i] = 1

M1 = M

M2 = np.zeros((N, N))
M2[0, 0] = 4 / 3
M2[0, 1] = -1 / 3

M3 = np.zeros((N, N))
M3[N - 1, N - 2] = -1 / 3
M3[N - 1, N - 1] = 4 / 3

M = M1 + M2 + M3
M = M / (dx ** 2)

D, V = eigs(- M, k=5, which='SM')

D = D[:5]
V = V[:, :5]

# boundry conditions
phi_0 = (4 / 3) * V[0, :] - (1 / 3) * V[1, :]
phi_n = - (1 / 3) * V[-2, :] + (4 / 3) * V[-1, :]

V = np.vstack((phi_0, V, phi_n))

# normalize
for i in range(5):
    norm = np.trapezoid(V[:, i] ** 2, x)
    V[:, i] = abs(V[:, i] / np.sqrt(norm))
    plt.plot(x, V[:, i])

plt.legend(["$\\phi_1$", "$\\phi_2$", "$\\phi_3$", "$\\phi_4$", "$\\phi_5$"], loc="upper right")
plt.xlabel("x")
plt.ylabel("Eigenfunctions")
plt.title("First Five Normalized Eigenfunctions")
plt.grid()

E_solb = D
y_solb = V
A3 = V
A4 = D

print('A3 = \n', A3)
print('A4 = \n', A4)

plt.show()

# Part c
L = 2
K = 1
dx = 0.1
tol = 1e-6
xshoot = np.arange(-L, L + dx, dx)
gamma_values = [0.05, - 0.05]

A5, A7 = np.zeros((len(xshoot), 2)), np.zeros((len(xshoot), 2))
A6, A8 = np.zeros(2), np.zeros(2)


def shoot2(x, phi, epsilon, gamma):
    return [phi[1],
            (gamma * abs(phi[0]) ** 2 + K * x ** 2 - epsilon) * phi[0]]


for gamma in gamma_values:
    epsilon_start = -1
    A = 1e-6
    for modes in range(1, 3):
        dA = 0.01
        for k in range(100):
            epsilon = epsilon_start
            depsilon = 0.2
            for kk in range(100):
                phi0 = [A, np.sqrt(K * L ** 2 - epsilon) * A]
                ans = solve_ivp(
                    lambda x, phi: shoot2(x, phi, epsilon, gamma),
                    [xshoot[0], xshoot[-1]],
                    phi0,
                    t_eval=xshoot
                )
                phi_sol = ans.y.T
                x_sol = ans.t
                bc = phi_sol[-1, 1] + np.sqrt(L ** 2 - epsilon) * phi_sol[-1, 0]
                if abs(bc) < tol:
                    break
                if (-1) ** (modes + 1) * bc > 0:
                    epsilon += depsilon
                else:
                    epsilon -= depsilon
                    depsilon /= 2

            # check if it is focused
            integral = np.trapezoid(phi_sol[:, 0] ** 2, x=x_sol)
            if abs(integral - 1) < tol:
                break
            if integral < 1:
                A += dA
            else:
                A -= dA
                dA /= 2

        epsilon_start = epsilon + 0.2

        if gamma > 0:
            A5[:, modes - 1] = np.abs(phi_sol[:, 0])
            A6[modes - 1] = epsilon

        else:
            A7[:, modes - 1] = np.abs(phi_sol[:, 0])
            A8[modes - 1] = epsilon

plt.plot(xshoot, A5)
plt.plot(xshoot, A7)
plt.legend(["$\\phi_1$", "$\\phi_2$"], loc="upper right")
plt.show()

print('A5 = \n', A5)
print('A6 = \n', A6)
print('A7 = \n', A7)
print('A8 = \n', A8)


# part d)
def hw1_rhs_a(x, y, E):
    return [y[1], (x ** 2 - E) * y[0]]


L = 2
x_span = [-L, L]
E = 1
A = 1
y0 = [A, np.sqrt(L ** 2 - E) * A]
tols = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
dt45, dt23, dt_radau, dt_bdf = [], [], [], []

for tol in tols:
    options = {'rtol': tol, 'atol': tol}

    sol45 = solve_ivp(hw1_rhs_a, x_span, y0, method='RK45', args=(E,), **options)
    sol23 = solve_ivp(hw1_rhs_a, x_span, y0, method='RK23', args=(E,), **options)
    sol_radau = solve_ivp(hw1_rhs_a, x_span, y0, method='Radau', args=(E,), **options)
    sol_bdf = solve_ivp(hw1_rhs_a, x_span, y0, method='BDF', args=(E,), **options)

    dt45.append(np.mean(np.diff(sol45.t)))
    dt23.append(np.mean(np.diff(sol23.t)))
    dt_radau.append(np.mean(np.diff(sol_radau.t)))
    dt_bdf.append(np.mean(np.diff(sol_bdf.t)))

fit45 = np.polyfit(np.log(dt45), np.log(tols), 1)
fit23 = np.polyfit(np.log(dt23), np.log(tols), 1)
fit_radau = np.polyfit(np.log(dt_radau), np.log(tols), 1)
fit_bdf = np.polyfit(np.log(dt_bdf), np.log(tols), 1)

slopes = [float(fit45[0]), float(fit23[0]), float(fit_radau[0]), float(fit_bdf[0])]

A9 = slopes
print('A9 = \n', A9)


# part e)
h = np.array([np.ones_like(x), 2 * x, 4 * (x ** 2) - 2, 8 * (x ** 3) - 12 * x, 16 * (x ** 4) - 48 * (x ** 2) + 12])
L = 4
dx = 0.1
x = np.arange(-L, L + dx, dx)
phi = np.zeros((len(x), 5))

def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

for j in range(5):
    phi[:, j] = (np.exp(-(x ** 2) / 2) * h[j, :] / np.sqrt(factorial(j) * (2 ** j) * np.sqrt(np.pi))).T

erpsi_a = np.zeros(5)
erpsi_b = np.zeros(5)
er_a = np.zeros(5)
er_b = np.zeros(5)

for i in range(5):
    erpsi_a[i] = np.trapezoid((abs(y_sola[:, i]) - abs(phi[:, i])) ** 2, x=x)
    erpsi_b[i] = np.trapezoid((abs(y_solb[:, i]) - abs(phi[:, i])) ** 2, x=x)

    er_a[i] = 100 * abs(E_sola[i] - (2 * (i + 1) - 1)) / (2 * (i + 1) - 1)
    er_b[i] = 100 * abs(E_solb[i] - (2 * (i + 1) - 1)) / (2 * (i + 1) - 1)

A10 = erpsi_a
A12 = erpsi_b
A11 = er_a
A13 = er_b

print('A10 = \n', A10)
print('A11 = \n', A11)
print('A12 = \n', A12)
print('A13 = \n', A13)