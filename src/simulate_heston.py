"""
simulate_heston.py 

This scripts simulates Heston's 93 model data, returns and integrated variance of the simulation. 
"""

##################
### 1. Imports ###
##################

import warnings 
import numpy as np 
from typing import Tuple, Optional

##################
### 2. Helpers ###
##################

def validate_parameters(kappa: float, theta: float, sigma: float) -> bool:
    """
    Check if the Heston model parameters are valid.

    Parameters:
    kappa (float): Mean reversion rate.
    theta (float): Long-term mean of variance.
    sigma (float): Volatility of volatility.

    Returns:
    bool: True if parameters are valid, False otherwise.
    """
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


def simulate_volatility_for_day(kappa: float, theta: float, sigma: float, delta: float, vol_past: float) -> float:
    """
    Simulate the volatility path for a single day.

    Parameters:
    kappa (float): Mean reversion rate.
    theta (float): Long-term mean of variance.
    sigma (float): Volatility of volatility.
    delta (float): Time step.
    vol_past (float): Previous day's volatility.

    Returns:
    float: Simulated volatility for the day.
    """
    vol_new = max(vol_past + kappa * (theta - vol_past) * delta + 
                  sigma * np.sqrt(vol_past * delta) * np.random.normal(0, 1), 0.000001)
    
    return vol_new

##############################
### 3. Simulation function ###
##############################

def compute_daily_return(mu: float, kappa: float, theta: float, sigma: float, rho: float, delta: float, 
                         V_today: float, V_yesterday: float, IV_today: float) -> float:
    """
    Compute the daily return.

    Parameters:
    mu (float): Drift rate of the stock price.
    kappa (float): Mean reversion rate.
    theta (float): Long-term mean of variance.
    sigma (float): Volatility of volatility.
    rho (float): Correlation between stock price and variance.
    delta (float): Time step.
    V_today (float): Today's volatility.
    V_yesterday (float): Yesterday's volatility.
    IV_today (float): Today's integrated variance.

    Returns:
    float: Simulated return for the day.
    """
    return (mu * delta - IV_today / 2 + rho / sigma * 
            (V_today - V_yesterday - kappa * theta * delta + kappa * IV_today) + 
            np.random.normal(0, np.sqrt((1 - rho ** 2) * IV_today)))


def simulate_heston93(mu: float, kappa: float, theta: float, sigma: float, rho: float, v_start: float, 
                      num_days: int, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate returns and volatility paths for the Heston model over a number of days.

    Parameters:
    - mu (float): Drift rate of the stock price.
    - kappa (float): Mean reversion rate.
    - theta (float): Long-term mean of variance.
    - sigma (float): Volatility of volatility.
    - rho (float): Correlation between stock price and variance.
    - v_start (float): Initial volatility.
    - num_days (int): Number of days to simulate.
    - seed (int, optional): Seed for the random number generator.

    Returns:
    - Tuple[np.ndarray, np.ndarray, np.ndarray]: Arrays of simulated returns, end-of-day volatilities,
                                                 and integrated variances.
    """
    # Validate the input parameters for the Heston model
    if not validate_parameters(kappa, theta, sigma):
        return None, None, None
    
    # Set the random seed if provided, for reproducibility
    if seed is not None:
        np.random.seed(seed)

    # Define the time step for one trading day
    delta = 1/252

    # Initialize arrays to store the results: volatility, returns, and integrated variance
    V = np.zeros(num_days + 1)  # Array for volatility, with initial value included
    ret = np.zeros(num_days)    # Array for daily returns
    IV = np.zeros(num_days)     # Array for integrated variance

    # Set the initial volatility
    V[0] = v_start

    # Loop over each day to simulate the model
    for t in range(num_days):
        # Start with the previous day's volatility
        V_today = V[t]

        # Simulate intraday volatility steps
        for _ in range(5):  # Each trading day is divided into 5 intraday steps
            V_today = simulate_volatility_for_day(kappa, theta, sigma, delta/5, V_today)
        
        # Update the volatility for the current day
        V[t + 1] = V_today

        # Calculate the integrated variance for the current day
        IV[t] = (V[t + 1] + V[t]) / 2 * delta

        # Compute the return for the current day
        ret[t] = compute_daily_return(mu, kappa, theta, sigma, rho, delta, V[t + 1], V[t], IV[t])

        # Add noise to the integrated variance to represent measurement error
        IV[t] += np.random.normal(0, 0.1 * IV[t])

    # Return the simulated returns, volatilities (excluding the initial value), and integrated variances
    return ret, V[1:], IV


