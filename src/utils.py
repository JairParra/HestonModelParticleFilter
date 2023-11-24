"""
utils.py 

Provide a number utility functions for other modules. 

@author: Hair Albeiro Parra Barrera
"""
################################################################################
##################
### 1. Imports ###
##################

# general imports
import os
import pprint 
import concurrent
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm 
from concurrent.futures import ProcessPoolExecutor


# custom modules 
from src.particle_filter import heston93_particle_filter 

## configurations 

# setup pretty printing
pp = pprint.PrettyPrinter(indent = 4)

# setup dark plotting
sns.set(style="darkgrid", context="talk")
plt.style.use("dark_background")
plt.rcParams.update({"grid.linewidth":0.5, "grid.alpha":0.5})


##################
### 2. Helpers ###
##################

def calculate_log_likelihoods_for_params(base_params, param_ranges, simul, ret, rv=None):
    """
    Calculate log-likelihoods for a range of values for each parameter in a given model.

    Args:
    base_params (list): Base parameter values for the model.
    param_ranges (dict): Dictionary of parameter names and their ranges to test.
    simul (dict): Simulated noise data.
    ret (np.array): Array of returns.
    rv (np.array, optional): Array of realized variances, if available. Defaults to None.

    Returns:
    dict: Dictionary containing parameter names as keys and tuples of parameter values and 
          corresponding log-likelihoods as values.
    """
    results = {}
    # Loop over each parameter and its range
    for i, param_name in tqdm(enumerate(param_ranges.keys()), 
                              desc="fitting Heston for params...", 
                              total=len(param_ranges.keys())):
        print(f"\n\t(Varying parameter: {param_name})")
        
        L = []  # List to store log-likelihoods for this parameter
        for value in tqdm(param_ranges[param_name], desc="fitting ..."):
            
            x = base_params.copy()
            x[i] = value  # Update the specific parameter value
            
            # Calculate log-likelihood depending on whether rv is provided
            if rv is not None:
                likelihood, _ = heston93_particle_filter(x, simul, 
                                                          np.column_stack((ret, rv)),
                                                          1, transform_params=False, 
                                                          verbose=False)
            else:
                likelihood, _ = heston93_particle_filter(x, simul, ret, 1, 
                                                          transform_params=False,
                                                          verbose=False)
            L.append(likelihood)
        # Store the parameter values and their corresponding log-likelihoods
        results[param_name] = (param_ranges[param_name], L)
    return results
