"""
simulation.py 

This script simulates Heston's model and performs a variety of analysis. 
Based on the original matlab script provided for the assignment. 

@author: Hair Albeiro Parra Barrera
"""

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
from scipy.optimize import fmin, fmin_bfgs

# simulation and filtering scripts 
from src.heston import simulate_heston93
from src.heston import heston93_particle_filter 
from src.utils import calculate_log_likelihoods_for_params

# custom plotting  
from src.plotting import plot_simulations
from src.plotting import plot_filtered_variance
from src.plotting import plot_returns_analysis
from src.plotting import plot_returns_rv_analysis
from src.plotting import plot_log_likelihood_ret
from src.plotting import plot_log_likelihood_ret_rv

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
eta = 0.2
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
eta_transformed = np.log(eta)
tparams = [mu_transformed, kappa_transformed, theta_transformed, sigma_transformed, rho_transformed, eta_transformed]

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
Like_H_true_ret, V_ret_true = heston93_particle_filter(tparams, simul, ret, 1, verbose=True)

# Filter using the returns, the Realized Variance and the true parameters
start_time = time.time()
Like_H_true_rv, V_RV_true = heston93_particle_filter(tparams, simul, np.column_stack((ret, RV)), 1, verbose=True)
end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")


# Call the function with the filtered variance data
plot_filtered_variance(V_RV_true, num_days)

################################
### 4. Estimating Parmaeters ###
################################

## Note: set verbose=False to avoid the output at every iteration

# set to False to avoid fittig from scratch
reoptimize = True

if reoptimize: 
    # Transformed parameters
    start_time = time.time()
    opt_ret = fmin(
        func=lambda x: heston93_particle_filter(np.concatenate([x, [eta_transformed]]),
                                                simul, ret, 1,
                                                verbose=True)[0],  # Only use the log likelihood
        x0=tparams[:5],
        xtol=1e-2, 
        ftol=1e-2, 
        maxiter=200, # I put a limit since it doesn't seem to be converging 
        disp=True   # Set to False to turn off the iteration messages
    )
    end_time = time.time() # might take ~1min per 10 steps
    print(f"Execution time: {end_time - start_time} seconds")
    
    # save the optimal parameters at this step 
    with open('data_clean/opt_params/opt_ret.npy', 'wb') as f: 
        np.save(f, opt_ret)
    
    # all parameters
    start_time = time.time()
    opt_rv = fmin(
        func=lambda x: heston93_particle_filter(x, simul, 
                                                np.column_stack((ret, RV)),
                                                1, verbose=False)[0], # Only use the log likelihood
        x0=tparams,
        xtol=1e-2,
        ftol=1e-2,
        maxiter=200, # I put a limit since it doesn't seem to be converging 
        disp=True
    )
    print(f"Execution time: {end_time - start_time} seconds")
    
    # save the optimal parameters at this step 
    with open('data_clean/opt_params/opt_rv.npy', 'wb') as f: 
        np.save(f, opt_rv)

    # save the optimal parameters at this step 
    with open('data_clean/opt_params/opt_rv.npy', 'wb') as f: 
        np.save(f, opt_rv)
        
else: 
    # load pre-optimized parameters
    opt_ret = np.load('data_clean/opt_params/opt_ret.npy')
    opt_rv = np.load('data_clean/opt_params/opt_rv.npy')
    

# Compute the filtered volatility
Like_H_ret, V_ret_est = heston93_particle_filter(opt_ret, simul, ret, 1, verbose=True)
Like_H_rv, V_RV_est = heston93_particle_filter(opt_rv, simul, np.column_stack((ret, RV)), 1, verbose=True)

# Call the function for plotting analysis using only returns
plot_returns_analysis(ret, V, V_ret_true, V_ret_est, num_days)

# Call the function for plotting analysis using returns and realized variance
plot_returns_rv_analysis(ret, V, V_RV_true, V_RV_est, num_days)

# Plot fitered variance with returns sample, optimized parameters
plot_filtered_variance(V_ret_est, num_days, spec="Returns Sample")

# Plot fitered variance with returns sample and RV, optimized parameters
plot_filtered_variance(V_RV_est, num_days, spec="Returns + RV Sample")

# Print the original parameters vs optimized parameters
print(f"True params: mu={mu}, kappa={kappa}, theta={theta}, rho={rho}, sigma={sigma}, eta={eta}")
print(f"opt_ret: mu={opt_ret[0]}, kappa={opt_ret[1]}, theta={opt_ret[2]}, rho={opt_ret[3]}, sigma={opt_ret[4]}")
print(f"opt_rv: mu={opt_rv[0]}, kappa={opt_rv[1]}, theta={opt_ret[2]}, rho={opt_rv[3]}, sigma={opt_rv[4]}, eta={opt_rv[5]}")

#######################################
### 5. Histograms Real vs Fitted Vt ###
#######################################

# Want to create histograms of the errors 


##############################
### 6. Likelihood Analysis ###
##############################

# Optimal parameters obtained from previous optimization
opt_ret_tr = [mu_transformed, kappa_transformed, theta_transformed, sigma_transformed, rho_transformed]
opt_rv_tr = [mu_transformed, kappa_transformed, theta_transformed, sigma_transformed, rho_transformed, eta_transformed]

# Define ranges for parameters for returns only
param_ranges_ret = {
    'mu': np.linspace(0.06 - 0.2, 0.06 + 0.2, 50),
    'kappa': np.linspace(5 - 3, 5 + 3, 50),
    'theta': np.linspace(0.02 - 0.01, 0.02 + 0.015, 50),
    'sigma': np.linspace(0.3 - 0.1, 0.3 + 0.1, 50),
    'rho': np.linspace(-0.7 - 0.2, -0.7 + 0.2, 50)
}

# Define ranges for returns and RV 
param_ranges_rv = param_ranges_ret.copy()
param_ranges_rv['eta'] = np.linspace(0.2 - 0.1, 0.2 + 0.1, 50)

# Extract optimized parameters for returns only
mu_opt_ret = opt_ret_tr[0] / 10
kappa_opt_ret = np.exp(opt_ret_tr[1])
theta_opt_ret = np.exp(opt_ret_tr[2])
sigma_opt_ret = np.exp(opt_ret_tr[3])
rho_opt_ret = (1 - np.exp(-opt_ret_tr[4])) / (1 + np.exp(-opt_ret_tr[4]))

# Similarly for returns and RV
mu_opt_rv = opt_rv_tr[0] / 10
kappa_opt_rv = np.exp(opt_rv_tr[1])
theta_opt_rv = np.exp(opt_rv_tr[2])
sigma_opt_rv = np.exp(opt_rv_tr[3])
rho_opt_rv = (1 - np.exp(-opt_rv_tr[4])) / (1 + np.exp(-opt_rv_tr[4]))
eta_opt_rv = np.exp(opt_rv_tr[5])

# repack transformed parameters 
opt_ret = [mu_opt_ret, kappa_opt_ret, theta_opt_ret, sigma_opt_ret, rho_opt_ret]
opt_rv = [mu_opt_rv, kappa_opt_rv, theta_opt_rv, sigma_opt_rv, rho_opt_rv, eta_opt_rv]

# Calculate log-likelihoods for each parameter range
results_ret = calculate_log_likelihoods_for_params(opt_ret, param_ranges_ret, simul, ret)
results_rv = calculate_log_likelihoods_for_params(opt_rv, param_ranges_rv, simul, ret, RV)

# Plotting for Returns Only
plot_log_likelihood_ret(results_ret)

# Plotting for Returns and RV
plot_log_likelihood_ret_rv(results_rv)
