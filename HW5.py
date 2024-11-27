import numpy as np
from scipy.fftpack import fft2, ifft2
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.sparse import spdiags, csr_matrix
from scipy.linalg import lu, solve_triangular
import time
from scipy.sparse.linalg import bicgstab, gmres
from matplotlib.animation import FuncAnimation

#part a)
# Define parameters
tspan = np.arange(0, 4.5, 0.5)
nu = 0.001
Lx, Ly = 20, 20
nx, ny = 64, 64
N = nx * ny

# Define spatial domain and initial conditions
x2 = np.linspace(-Lx/2, Lx/2, nx + 1)
x = x2[:nx]
y2 = np.linspace(-Ly/2, Ly/2, ny + 1)
y = y2[:ny]
X, Y = np.meshgrid(x, y)

w = np.exp(- X**2 - Y**2/20)
w2 = w.reshape(N)

# Define spectral k values
kx = (2 * np.pi / Lx) * np.concatenate((np.arange(0, nx/2), np.arange(-nx/2, 0)))
ky = (2 * np.pi / Ly) * np.concatenate((np.arange(0, ny/2), np.arange(-ny/2, 0)))
KX, KY = np.meshgrid(kx, ky)
K = KX**2 + KY**2
K[0, 0] = 1e-6  # Avoid division by zero

m = 64   # N value in x and y directions
n = m * m  # total size of matrix
dx = 20/m

e0 = np.zeros((n, 1))  # vector of zeros
e1 = np.ones((n, 1))   # vector of ones
e2 = np.copy(e1)    # copy the one vector
e4 = np.copy(e0)    # copy the zero vector

for j in range(1, m+1):
    e2[m*j-1] = 0  # overwrite every m^th value with zero
    e4[m*j-1] = 1  # overwirte every m^th value with one

# Shift to correct positions
e3 = np.zeros_like(e2)
e3[1:n] = e2[0:n-1]
e3[0] = e2[n-1]

e5 = np.zeros_like(e4)
e5[1:n] = e4[0:n-1]
e5[0] = e4[n-1]

# Place diagonal elements
diagonals_A = [e1.flatten(), e1.flatten(), e5.flatten(),
             e2.flatten(), -4 * e1.flatten(), e3.flatten(),
             e4.flatten(), e1.flatten(), e1.flatten()]
offsets_A = [-(n-m), -m, -m+1, -1, 0, 1, m-1, m, (n-m)]
A = spdiags(diagonals_A, offsets_A, n, n).toarray() / dx**2
A[0,0] = 2 / dx**2

diagonals_B = [e1.flatten(), -e1.flatten(), e1.flatten(), -e1.flatten()]
offsets_B = [-(n-m), -m, m, (n-m)]
B = spdiags(diagonals_B, offsets_B, n, n).toarray() / (2*dx)

diagonals_C = [e5.flatten(), -e2.flatten(), e3.flatten(), -e4.flatten()]
offsets_C = [-m+1, -1, 1, m-1]
C = spdiags(diagonals_C, offsets_C, n, n).toarray() / (2*dx)

# Define the ODE system
def spc_rhs(t, w2, nx, ny, N, A, B, C, K, nu):
    w = w2.reshape((nx, ny))
    wt = fft2(w)
    psit = -wt / K
    psi = np.real(ifft2(psit)).reshape(N)
    rhs = nu * np.dot(A, w2) + (np.dot(B, w2)) * (np.dot(C, psi)) - (np.dot(B, psi)) * (np.dot(C, w2))
    return rhs

#Time Recoding
start_time = time.time() # Record the start time
wsol = solve_ivp(spc_rhs, [0,4], w2, t_eval=tspan, args=(nx, ny, N, A, B, C, K, nu), method='RK45')
end_time = time.time()  # Record the end time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")
A1 = wsol.y
print(f"A1 = {A1}")

# Plot A1
sol_to_plot = A1
n = int (np.sqrt (sol_to_plot.shape[0]))  # n = sqrt(4096)
fig, ax = plt.subplots (figsize=(6, 6))
cax = ax.imshow (sol_to_plot[:, 0].reshape ((n, n)), extent=[-10, 10, -10, 10], cmap='jet')
fig.colorbar (cax, ax=ax, label='Vorticity')
ax.set_title ('Vorticity Field - FFT')
ax.set_xlabel ('x')
ax.set_ylabel ('y')
def update (frame):
     ax.set_title (f'Vorticity Field at t = {frame * 0.5:.2f}')
     cax.set_data (sol_to_plot[:, frame].reshape ((n, n)))
     return cax,
anim = FuncAnimation (fig, update, frames=sol_to_plot.shape[1], blit=True)
anim.save('/Users/alicexxi_/Desktop/学习/AMATH481/vorticity_evolution_FFT.gif', writer='pillow', fps=2)


# Part b)
# A/b
def GE_rhs(t,w2, nx, ny, N, A, B, C, K, nu):
    psi= np.linalg.solve(A,w2)
    rhs=nu*np.dot(A,w2)+(np.dot(B,w2))*(np.dot(C,psi))-(np.dot(B,psi))*(np.dot(C,w2))
    return rhs

# Time Recoding
start_time = time.time() # Record the start time
wtsol = solve_ivp(GE_rhs, [0, 4], w2, t_eval=tspan, args=(nx, ny, N, A, B, C, K, nu), method='RK45')
end_time = time.time()  # Record the end time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")
A2=wtsol.y
print(f"A2 = {A2}")

# Plot A2
sol_to_plot = A2
n = int (np.sqrt (sol_to_plot.shape[0]))  # n = sqrt(4096)
fig, ax = plt.subplots (figsize=(6, 6))
cax = ax.imshow (sol_to_plot[:, 0].reshape ((n, n)), extent=[-10, 10, -10, 10], cmap='jet')
fig.colorbar (cax, ax=ax, label='Vorticity')
ax.set_title ('Vorticity Field - A/b')
ax.set_xlabel ('x')
ax.set_ylabel ('y')
def update (frame):
    ax.set_title (f'Vorticity Field at t = {frame * 0.5:.2f}')
    cax.set_data (sol_to_plot[:, frame].reshape ((n, n)))
    return cax,
anim = FuncAnimation (fig, update, frames=sol_to_plot.shape[1], blit=True)
anim.save('/Users/alicexxi_/Desktop/学习/AMATH481/vorticity_evolution_Ab.gif', writer='pillow', fps=2)


#LU
P, L, U = lu(A)
def LU_rhs(t,w2, nx, ny, N, A, B, C, K,nu):
    Pb=np.dot(P,w2)
    y=solve_triangular(L,Pb,lower=True)
    psi=solve_triangular(U,y)
    rhs=nu*np.dot(A,w2)+(np.dot(B,w2))*(np.dot(C,psi))-(np.dot(B,psi))*(np.dot(C,w2))
    return rhs

# Time Recoding
start_time = time.time() # Record the start time
wtsol = solve_ivp(LU_rhs, [0, 4], w2, t_eval=tspan, args=(nx, ny, N, A, B, C, K, nu), method='RK45')
end_time = time.time()  # Record the end time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")
A3=wtsol.y
print('A3 = \n', A3)

sol_to_plot = A3
n = int (np.sqrt (sol_to_plot.shape[0]))  # n = sqrt(4096)
fig, ax = plt.subplots (figsize=(6, 6))
cax = ax.imshow (sol_to_plot[:, 0].reshape ((n, n)), extent=[-10, 10, -10, 10], cmap='jet')
fig.colorbar (cax, ax=ax, label='Vorticity')
ax.set_title ('Vorticity Field - A/b')
ax.set_xlabel ('x')
ax.set_ylabel ('y')
def update (frame):
    ax.set_title (f'Vorticity Field at t = {frame * 0.5:.2f}')
    cax.set_data (sol_to_plot[:, frame].reshape ((n, n)))
    return cax,
anim = FuncAnimation (fig, update, frames=sol_to_plot.shape[1], blit=True)
anim.save('/Users/alicexxi_/Desktop/学习/AMATH481/vorticity_evolution_LU.gif', writer='pillow', fps=2)


#BICGSTAB
A_sparse = csr_matrix (A)

def bicgstab_rhs (t, w, A, B, C, nu):
    psi, info = bicgstab (A_sparse, w, atol=1e-8, maxiter=1000)
    rhs = (nu * A.dot (w) + (B.dot (w)) * (C.dot (psi)) - (B.dot (psi)) * (C.dot (w)))
    return rhs

# Time Recoding
start_time = time.time ()
sol = solve_ivp (bicgstab_rhs,(tspan[0], tspan[-1]), w2, t_eval=tspan,args=(A, B, C, nu),method='RK45')
end_time = time.time ()
elapsed_time = end_time - start_time
print (f"Elapsed time for BICGSTAB: {elapsed_time:.2f} seconds")
wtsol_bicgstab = sol.y
A4 = wtsol_bicgstab
print ("A4", A4)

sol_to_plot = A4
n = int (np.sqrt (sol_to_plot.shape[0]))
fig, ax = plt.subplots (figsize=(6, 6))
cax = ax.imshow (sol_to_plot[:, 0].reshape ((n, n)), extent=[-10, 10, -10, 10], cmap='jet')
fig.colorbar (cax, ax=ax, label='Vorticity')
ax.set_title ('Vorticity Field - BICGSTAB')
ax.set_xlabel ('x')
ax.set_ylabel ('y')
def update (frame):
    ax.set_title (f'Vorticity Field at t = {frame * 0.5:.2f}')
    cax.set_data (sol_to_plot[:, frame].reshape ((n, n)))
    return cax,
anim = FuncAnimation (fig, update, frames=sol_to_plot.shape[1], blit=True)
anim.save('/Users/alicexxi_/Desktop/学习/AMATH481/vorticity_evolution_BICGSTAB.gif', writer='pillow', fps=2)


#GMRES
A_sparse = csr_matrix (A)

def gmres_rhs(t, w, A, B, C, nu):
    psi, info = gmres (A_sparse, w, atol=1e-8, restart=50, maxiter=1000)
    rhs = (nu * A.dot (w) + (B.dot (w)) * (C.dot (psi)) - (B.dot (psi)) * (C.dot (w)))
    return rhs

# Time Recoding
start_time = time.time ()
sol = solve_ivp (gmres_rhs,(tspan[0], tspan[-1]), w2, t_eval=tspan, args=(A, B, C, nu),method='RK45')
end_time = time.time ()
elapsed_time = end_time - start_time
print (f"Elapsed time for GMRES: {elapsed_time:.2f} seconds")
wtsol_gmres = sol.y
A5 = wtsol_gmres
print("A5", A5)

sol_to_plot = A5
n = int (np.sqrt (sol_to_plot.shape[0]))  # n = sqrt(4096)
fig, ax = plt.subplots (figsize=(6, 6))
cax = ax.imshow (sol_to_plot[:, 0].reshape ((n, n)), extent=[-10, 10, -10, 10], cmap='jet')
fig.colorbar (cax, ax=ax, label='Vorticity')
ax.set_title ('Vorticity Field - GMRES')
ax.set_xlabel ('x')
ax.set_ylabel ('y')
def update (frame):
    ax.set_title (f'Vorticity Field at t = {frame * 0.5:.2f}')
    cax.set_data (sol_to_plot[:, frame].reshape ((n, n)))
    return cax,
anim = FuncAnimation (fig, update, frames=sol_to_plot.shape[1], blit=True)
anim.save('/Users/alicexxi_/Desktop/学习/AMATH481/vorticity_evolution_GMRES.gif', writer='pillow', fps=2)

######################## Part C & D #######################

from scipy.fft import fft2, ifft2, fftfreq

# Parameters
L = 20  # Domain length [-L, L]
n = 64  # Number of grid points
x = np.linspace(-L, L, n)
y = np.linspace(-L, L, n)
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y)
nu = 0.001  # Diffusion coefficient
tspan = np.arange(0, 4.5, 0.5)

# Helper function for computing streamfunction
def compute_psi(omega):
    kx = fftfreq(n, d=dx) * 2 * np.pi
    ky = fftfreq(n, d=dy) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX**2 + KY**2
    K2[0, 0] = 1e-6  # Avoid division by zero
    psi_hat = fft2(omega) / K2
    return np.real(ifft2(psi_hat))

# Vorticity advection term
def vorticity_advection(omega, psi):
    psi_x = np.gradient(psi, dx, axis=1)
    psi_y = np.gradient(psi, dy, axis=0)
    omega_x = np.gradient(omega, dx, axis=1)
    omega_y = np.gradient(omega, dy, axis=0)
    return psi_x * omega_y - psi_y * omega_x

# RHS for time integration
def rhs(t, omega_flat):
    omega = omega_flat.reshape((n, n))
    psi = compute_psi(omega)
    adv = vorticity_advection(omega, psi)
    diff = nu * (np.gradient(np.gradient(omega, dx, axis=1), dx, axis=1) +
                 np.gradient(np.gradient(omega, dy, axis=0), dy, axis=0))
    return (diff - adv).flatten()

# Initial conditions for different cases
def initial_conditions(case):
    if case == "opposite_gaussians":
        return np.exp(-((X - 5)**2 + Y**2)) - np.exp(-((X + 5)**2 + Y**2))
    elif case == "same_gaussians":
        return np.exp(-((X - 5)**2 + Y**2)) + np.exp(-((X + 5)**2 + Y**2))
    elif case == "colliding_pairs":
        return (np.exp(-((X - 5)**2 + (Y - 5)**2)) - np.exp(-((X + 5)**2 + (Y - 5)**2)) +
                np.exp(-((X - 5)**2 + (Y + 5)**2)) - np.exp(-((X + 5)**2 + (Y + 5)**2)))
    elif case == "random_vortices":
        omega = np.zeros_like(X)
        for _ in range(10):
            x0, y0 = np.random.uniform(-L, L, size=2)
            strength = np.random.uniform(-1, 1)
            ellipticity = np.random.uniform(1, 5)
            omega += strength * np.exp(-((X - x0)**2 + (Y - y0)**2 / ellipticity))
        return omega
    else:
        raise ValueError("Invalid case")

# Solve for each case
case = "random_vortices"  # Choose from 'opposite_gaussians', 'same_gaussians', 'colliding_pairs', 'random_vortices'
omega0 = initial_conditions(case).flatten()
sol = solve_ivp(rhs, [0, tspan[-1]], omega0, t_eval=tspan, method="RK45")

# Reshape solution for visualization
omega_sol = sol.y.T.reshape(-1, n, n)

# Part d
import matplotlib.animation as animation

# Define spatial domain
x2 = np.linspace(-Lx / 2, Lx / 2, nx + 1)
x = x2[:nx]
y2 = np.linspace(-Ly / 2, Ly / 2, ny + 1)
y = y2[:ny]
X, Y = np.meshgrid(x, y)

# Define initial conditions for different vortex configurations
initial_conditions = {
    "opposite_gaussians": lambda: np.exp(-((X - 5) ** 2 + Y ** 2) / 20) - np.exp(-((X + 5) ** 2 + Y ** 2) / 20),
    "same_gaussians": lambda: np.exp(-((X - 5) ** 2 + Y ** 2) / 20) + np.exp(-((X + 5) ** 2 + Y ** 2) / 20),
    "colliding_pairs": lambda: np.exp(-((X - 5) ** 2 + (Y - 5) ** 2) / 20) - np.exp(
        -((X + 5) ** 2 + (Y + 5) ** 2) / 20),
    "random_vortices": lambda: sum(((-1) ** np.random.randint(0, 2)) * np.exp(
        -((X - np.random.uniform(-10, 10)) ** 2 + (Y - np.random.uniform(-10, 10)) ** 2) / 20) for _ in range(10))
}

# Define spectral k values
kx = (2 * np.pi / Lx) * np.concatenate((np.arange(0, nx / 2), np.arange(-nx / 2, 0)))
kx[0] = 1e-6
ky = (2 * np.pi / Ly) * np.concatenate((np.arange(0, ny / 2), np.arange(-ny / 2, 0)))
ky[0] = 1e-6
KX, KY = np.meshgrid(kx, ky)
K = KX ** 2 + KY ** 2


def run_solver(initial_condition, label):
    w = initial_condition()
    w2 = w.reshape(N)

    start_time = time.time()
    wtsol = solve_ivp(spc_rhs, [0, 4], w2, t_eval=tspan, args=(nx, ny, N, A, B, C, K, nu), method='RK45')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for {label}: {elapsed_time:.2f} seconds")

# plot final state
fig, ax = plt.subplots(figsize=(6, 6))

def update_plot(i):
    ax.clear()
    state = wtsol.y[:, i].reshape((nx, ny))
    contour = ax.contourf(X, Y, state, levels=50, cmap='viridis')
    ax.set_title(f"Time: {tspan[i]:.2f} s")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    return contour

ani = animation.FuncAnimation(fig, update_plot, frames=len(tspan), interval=500)
ani.save("vorticity_dynamics.gif", writer="pillow", fps=2)
plt.show()