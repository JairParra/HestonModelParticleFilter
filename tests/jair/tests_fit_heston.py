"""
TODO 
"""
################################################################################
##################
### 1. Imports ###
##################

# general imports
import pprint 
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

# custom modules 
from tests.jair.utils import plot_simulations
from tests.jair.simulate_heston_v2 import simulate_heston93
from tests.jair.heston_particle_filter import heston93_particle_filter 

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

# Setup parameters 
mu = 0.1  # drift parmaeter
kappa = 1.5 # ? 
theta = 0.04 # mean reversal parameter 
sigma = 0.2 # long-term volatility 
rho = -0.7 # BM correlation parameter
v_start = 0.04 # ? 
num_days = 100 # number of simulation days 
seed = 420 # for random generation 

# Simulate data using the Heston model
ret, V, IV = simulate_heston93(mu, kappa, theta, sigma, rho, v_start, num_days, seed=seed)

# Combine returns and integrated variance for observations
obs = np.column_stack((ret, IV)) 

# Simulated noises for particle filter
simul = {
    'U': np.random.uniform(size=(5000, num_days)),
    'ZZ': np.random.normal(size=(5000, num_days, 5)),
    'UU': np.random.uniform(size=(5000, num_days))
}

# Transformed parameters for particle filter
x = np.array([mu * 10, np.log(kappa), np.log(theta), np.log(sigma), -np.log((1 + rho) / (1 - rho)), np.log(0.1)])

# Apply particle filter to simulated data
like_h, filtered_V = heston93_particle_filter(x, simul, obs, 1)

# Output log likelihood and filtered variance
print(f"Log Likelihood: {like_h}")
print(f"Filtered Variance (first few values): \n{filtered_V[:5]}")
