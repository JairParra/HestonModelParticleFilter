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

def validate_parameters(kappa, theta, sigma):
    """Check if the Heston model parameters are valid."""
    if theta <= 0:
        warnings.warn('Theta should be greater than 0')
        return False

    if sigma <= 0:
        warnings.warn('Sigma should be greater than 0')
        return False

    if kappa <= 0:
        warnings.warn('Kappa should be greater than 0')
        return False

    if (2 * kappa * theta) <= (sigma ** 2):
        warnings.warn('2 * kappa * theta should be greater than sigma^2')
        return False

    return True

def simulate_volatility_for_day(kappa, theta, sigma, delta, vol_past):
    """Simulate the volatility path for a single day."""
    vol_new = max(vol_past + kappa * (theta - vol_past) * delta + sigma * np.sqrt(vol_past * delta) * np.random.normal(0, 1), 0.000001)
    return vol_new

def compute_daily_return(mu, kappa, theta, sigma, rho, delta, V_today, V_yesterday, IV_today):
    """Compute the daily return."""
    return (mu * delta - IV_today / 2 + rho / sigma * (V_today - V_yesterday - kappa * theta * delta + kappa * IV_today) + 
            np.random.normal(0, np.sqrt((1 - rho ** 2) * IV_today)))

##############################
### 3. Simulation function ###
##############################

## Version 1 
def simulate_heston93v0(mu, kappa, theta, sigma, rho, v_start, num_days):
    
    # Check if the parameters are valid
    if theta <= 0:
        warnings.warn('Theta should be greater than 0')
        return None, None, None

    if sigma <= 0:
        warnings.warn('Sigma should be greater than 0')
        return None, None, None

    if kappa <= 0:
        warnings.warn('Kappa should be greater than 0')
        return None, None, None

    if (2 * kappa * theta) <= (sigma ** 2):
        warnings.warn('2 * kappa * theta should be greater than sigma^2')
        return None, None, None

    # Time step, using one trading day
    delta = 1/252

    # Preallocate the arrays that will store the volatility, returns, and integrated variance
    V = np.zeros(num_days + 1)  # +1 for the initial volatility
    ret = np.zeros(num_days)
    IV = np.zeros(num_days)
    
    # Set the initial volatility
    V[0] = v_start

    # Simulate each day
    for t in range(num_days):
        vol_past = V[t]
        intraday_vol = [vol_past]
        
        # Simulate the volatility within the day
        for d in range(5):  # 5 intraday steps
            vol_new = max(vol_past + kappa * (theta - vol_past) * delta / 5 + sigma * np.sqrt(vol_past * delta / 5) * np.random.normal(0, 1), 0.000001)
            intraday_vol.append(vol_new)
            vol_past = vol_new
        
        # Store the end of day volatility
        V[t + 1] = intraday_vol[-1]

        # Compute the integrated variance for the day
        IV[t] = np.sum((np.array(intraday_vol[:-1]) + np.array(intraday_vol[1:])) / 2) * delta / 5

        # Simulate the return for the day
        ret[t] = mu * delta - IV[t] / 2 + rho / sigma * (V[t + 1] - V[t] - kappa * theta * delta + kappa * IV[t]) + np.random.normal(0, np.sqrt((1 - rho ** 2) * IV[t]))

        # Add noise to the integrated variance to simulate measurement error
        IV[t] += np.random.normal(0, 0.1 * IV[t])

    # Note that V has num_days + 1 elements, ret and IV both have num_days elements
    return ret, V[1:], IV

################################################################################
## Version 2

def simulate_heston93v1(mu, kappa, theta, sigma, rho, v_start, num_days):
    if not validate_parameters(kappa, theta, sigma):
        return None, None, None

    # Time step for one trading day
    delta = 1/252

    # Arrays to store the results
    V = np.zeros(num_days + 1)
    ret = np.zeros(num_days)
    IV = np.zeros(num_days)
    
    V[0] = v_start  # Initial volatility

    for t in range(num_days):
        V_today = V[t]
        for _ in range(5):  # 5 intraday steps
            V_today = simulate_volatility_for_day(kappa, theta, sigma, delta/5, V_today)
        
        V[t + 1] = V_today
        IV[t] = (V[t + 1] + V[t]) / 2 * delta
        ret[t] = compute_daily_return(mu, kappa, theta, sigma, rho, delta, V[t + 1], V[t], IV[t])
        IV[t] += np.random.normal(0, 0.1 * IV[t])  # Measurement error

    return ret, V[1:], IV  # Note: V[0] is excluded to match dimensions with ret and IV

