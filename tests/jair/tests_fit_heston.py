"""
TODO 
"""
################################################################################
##################
### 1. Imports ###
##################

# general imports
import time
import pprint 
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

# computational  
from scipy.optimize import fmin 

# custom modules 
from src.utils import plot_simulations
from src.utils import plot_filtered_variance
from src.utils import plot_returns_analysis
from src.utils import plot_returns_rv_analysis
from src.simulate_heston import simulate_heston93
from src.particle_filter import heston93_particle_filter 
from src.particle_filter import heston93_particle_filter_nat

## configurations 

# setup pretty printing
pp = pprint.PrettyPrinter(indent = 4)

# setup dark plotting
sns.set(style="darkgrid", context="talk")
plt.style.use("dark_background")
plt.rcParams.update({"grid.linewidth":0.5, "grid.alpha":0.5})

################################################################################
#####################
### 2. Simulation ###
#####################

# Simulated observations
# Calibrate the number of days
num_days = 1000  # Number of days

# Chose a set of parameters
mu = 0.06  # drift
kappa = 5  # speed of mean reversion
theta = 0.02  # Mean reversion level
rho = -0.7  # Correlation
sigma = 0.3  # volatility of volatility
v = 0.01

# Simulation of returns
ret, V, RV = simulate_heston93(mu, kappa, theta, sigma, rho, v, num_days)

# Call the function with the simulated data
plot_simulations(ret, V, RV, num_days)

######################################
### 3. Filter with true parameters ###
######################################

# Transformed parameters
mu_transformed = mu * 10
sigma_transformed = np.log(sigma)
kappa_transformed = np.log(kappa)
theta_transformed = np.log(theta)
rho_transformed = np.log(1 + rho) - np.log(1 - rho)
eta_transformed = np.log(0.2)
x = [mu_transformed, kappa_transformed, theta_transformed, sigma_transformed, rho_transformed, eta_transformed]

# Generating the random variables
nb_part = 10000
N = len(ret)
simul = {
    'U': np.random.uniform(0, 1, (nb_part, 1)),
    'ZZ': np.random.normal(0, 1, (nb_part, num_days, 5)),
    'UU': np.random.uniform(0, 1, (nb_part, num_days)),
    'Z': np.random.normal(0, 1, (nb_part, num_days))
}

# Filter using only the returns and the true parameters
Like_H_true_ret, V_ret_true = heston93_particle_filter(x, simul, ret, 1)

# Filter using the returns, the Realized Variance and the true parameters
start_time = time.time()
Like_H_true_rv, V_RV_true = heston93_particle_filter(x, simul, np.column_stack((ret, RV)), 1)
end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")

###########################
### 4. Parameter Search ###
###########################

# Transformed parameters
start_time = time.time()
opt_ret = fmin(
    func=lambda y: heston93_particle_filter(np.concatenate([y, [eta_transformed]]),
                                            simul, ret, 1,
                                            verbose=False)[0],  # Only use the log likelihood
    x0=x[:5],
    xtol=1e-3,
    ftol=1e-3,
    maxiter=10, # I put a limit since it doesn't seem to be converging 
    disp=True  # Set to False to turn off the iteration messages
)
end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")


# all parameters
start_time = time.time()
opt_rv = fmin(
    func=lambda y: heston93_particle_filter(y, simul, 
                                            np.column_stack((ret, RV)),
                                            1, verbose=False)[0], # Only use the log likelihood
    x0=x,
    xtol=1e-3,
    ftol=1e-3,
    maxiter=10, # I put a limit since it doesn't seem to be converging 
    disp=True
)
print(f"Execution time: {end_time - start_time} seconds")

# Compute the filtered volatility
Like_H_ret, V_ret_est = heston93_particle_filter(opt_ret, simul, ret, 1)
Like_H_rv, V_RV_est = heston93_particle_filter(opt_rv, simul, np.column_stack((ret, RV)), 1)

# Call the function for plotting analysis using only returns
plot_returns_analysis(ret, V, V_ret_true, V_ret_est, num_days)

# Call the function for plotting analysis using returns and realized variance
plot_returns_rv_analysis(ret, V, V_RV_true, V_RV_est, num_days)


