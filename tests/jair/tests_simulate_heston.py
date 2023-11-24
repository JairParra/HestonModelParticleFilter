"""
simulate_heston.py 

This scripts simulates Heston's 93 model according to 
"""
################################################################################
##################
### 1. Imports ###
##################

# general imports
import pprint 
import warnings 
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

# custom modules 
from tests.jair.utils import plot_simulations
from tests.jair.simulate_heston_v0v1 import simulate_heston93v0 as sh0 
from tests.jair.simulate_heston_v0v1 import simulate_heston93v1 as sh1 
from tests.jair.simulate_heston_v2 import simulate_heston93 as sh2

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

### 1 Unoptimized Initial Heston Model 
ret, V, IV = sh0(mu, kappa, theta, sigma, rho, v_start, num_days)

# Call the function with the simulated data
plot_simulations(ret, V, IV, num_days)


### 2 Modularized version 
ret, V, IV = sh1(mu, kappa, theta, sigma, rho, v_start, num_days)

# Call the function with the simulated data
plot_simulations(ret, V, IV, num_days)


