import matplotlib
matplotlib.use('Agg') 
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def shoot(y, L, K, epsilon):
    return [y[1], (K * L ** 2 - epsilon) * y[0]]

tol = 1e-4  
col = ['r', 'b', 'g', 'c', 'm', 'k']  

L = 4; 
epsilon_start = 0.1; 
xp = [-L, L]; 
K = 1
xshoot = np.arange(xp[0], xp[1]+0.1, 0.1)
A1 = np.zeros((len(xshoot), 5))
A2 = []

for modes in range(1, 6):  
    epsilon = epsilon_start  
    depsilon = 0.2

    for _ in range(1000):  
        y0 = [1, np.sqrt(K * L ** 2 - epsilon)]; 
        y = odeint(shoot, y0, xshoot, args=(K,epsilon,)) 

        if abs(y[-1, 1] + np.sqrt(K * L**2 - epsilon) * y[-1, 0]) < tol:  
            break 

        if ((-1) ** (modes + 1) * (y[-1, 1] + np.sqrt(K * L**2 - epsilon) * y[-1, 0])) > 0:
            epsilon += depsilon
        else:
            epsilon -= depsilon 
            depsilon /= 2 
    A2.append(epsilon) 
    epsilon_start = epsilon + 0.1
    norm = np.trapz(y[:, 0] * y[:, 0], xshoot)
    eigenfuction = abs(y[:, 0] / np.sqrt(norm))
    A1[:, modes - 1] = eigenfuction
    
    plt.plot(xshoot, eigenfuction, col[modes - 1])  

plt.show() 
print(A1)
print(A2)