from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from gaussian_process import eval_u_t_np
from itertools import product
import scipy.stats


# Define the necessary parameters
N_SIM = 80   # number of simulations
N_F = 2      # number of frequencies
mu = 0.0
N_POINTS_P_AXIS = 80
TIME_STEP_LENGTH = 0.1 # it was 0.2
TOTAL_TIME = 2 # this was 10
N_TIME_STEPS = np.floor(TOTAL_TIME/TIME_STEP_LENGTH).astype(int)
time_vector = np.arange(0.0, TOTAL_TIME, TIME_STEP_LENGTH)


# Generate the gaussian BCs
np.random.seed(22)
truncate = False
mu = 0.0
sigma = 3 # 5

if truncate:
    lower = -10
    upper = 10
    N_samples = N_SIM*N_F
    coeff = np.array(scipy.stats.truncnorm.rvs((lower-mu)/sigma,(upper-mu)/sigma,loc=mu,scale=sigma,size=N_samples))
    alphaM = coeff.reshape((N_SIM, N_F))
else:
    alphaM = np.random.normal(mu, sigma, (N_SIM, N_F))


# Create a matrix that stores, in each row, the evaluations of u_i(t_j) for the i-th simulation
u_t_matrix = np.zeros((N_SIM, N_TIME_STEPS))
for i in range(N_SIM):
    u_t_matrix[i, :] = eval_u_t_np(time_vector, alphaM[i, :], TOTAL_TIME)

plt.rcParams.update({
        'axes.labelsize': 20,  
        'legend.fontsize': 18,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'figure.titlesize': 20 
    })

plt.figure(figsize=(8, 6))
for i in (4, 11, 53,65, 72): #range(1, N_SIM+1):
    alphav = alphaM[i-1, :]
    # Plot the u_t used for the i-th simulation over 2 periods
    t_v = np.linspace(0, TOTAL_TIME, 1000)
    u_plot = eval_u_t_np(t_v, alphav, TOTAL_TIME)
    plt.plot(t_v, u_plot)
plt.xlabel('$t$')
plt.ylabel('$v(t)$')
#plt.title('u(t) current simulation')
plt.grid(True, which='both', linestyle='--', alpha=0.2)
plt.savefig(f'PLOT.pdf')
plt.show()