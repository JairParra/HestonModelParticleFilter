"""
simulate_heston.py 

This scripts simulates Heston's 93 model according to 
"""

##################
### 1. Imports ###
##################

import warnings 
import numpy as np 

##################
### 2. Helpers ###
##################

def simulate_volatility_path(kappa, theta, sigma, delta, vol_past, steps_per_day):
    """Simulate the volatility path for one day using the Euler-Maruyama method."""
    # Using cumulative sum for vectorized computation of the volatility path
    dW = np.random.normal(0, np.sqrt(delta / steps_per_day), steps_per_day)
    v_increment = kappa * (theta - vol_past) * delta / steps_per_day + sigma * np.sqrt(vol_past * delta / steps_per_day) * dW
    v_path = np.maximum.accumulate(vol_past + np.cumsum(v_increment))
    v_path[0] = vol_past  # Ensure the first element is the previous day's closing volatility
    return v_path

def compute_integrated_variance(volatility_path, delta):
    """Compute the integrated variance for the day."""
    return np.sum((volatility_path[:-1] + volatility_path[1:]) / 2) * delta

##############################
### 3. Simulation function ###
##############################

def simulate_heston93(mu, kappa, theta, sigma, rho, v_start, num_days):
    """Simulate returns and volatility paths for the Heston model over a number of days."""
    
    # Validate model parameters
    if not all(param > 0 for param in [theta, sigma, kappa]) or (2 * kappa * theta) <= (sigma ** 2):
        warnings.warn('Model parameters are invalid.')
        return None, None, None

    # Trading day fraction for daily discretization
    delta = 1 / 252
    steps_per_day = 5  # Number of steps per day for intraday simulation

    # Preallocate arrays for results
    V = np.zeros(num_days + 1)
    ret = np.zeros(num_days)
    IV = np.zeros(num_days)

    # Start the simulation
    for t in range(num_days):
        vol_past = v_start if t == 0 else V[t]

        # Simulate the volatility path for the current day
        v_path = simulate_volatility_path(kappa, theta, sigma, delta, vol_past, steps_per_day)
        V[t + 1] = v_path[-1]
        
        # Compute the integrated variance for the day
        IV[t] = compute_integrated_variance(v_path, delta)
        
        # Simulate the log return for the day
        ret[t] = (mu * delta - IV[t] / 2 + rho / sigma * 
                       (V[t + 1] - vol_past - kappa * theta * delta + kappa * IV[t]) + 
                       np.random.normal(0, np.sqrt((1 - rho ** 2) * IV[t])))
        
        # Add noise to the integrated variance to represent measurement error
        IV[t] += np.random.normal(0, 0.1 * IV[t])

    return ret, V[1:], IV




