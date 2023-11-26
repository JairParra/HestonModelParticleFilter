"""
simulate_heston.py 

This scripts simulates Heston's 93 model data, returns and integrated variance of the simulation.
It also implements the particle filter for the Heston model.  

@author: Hair Albeiro Parra Barrera
"""

##################
### 1. Imports ###
##################

# general
import warnings 
import numpy as np 
from tqdm import tqdm 
from typing import Tuple, Optional

# stats
from scipy.stats import norm

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


##########################
### 4. Particle Filter ###
##########################

def heston93_particle_filter(x: np.ndarray, 
                             simul: dict, 
                             obs: np.ndarray, 
                             c_i: int, 
                             transform_params: bool = True, 
                             verbose : bool = False) -> Tuple[float, np.ndarray]:
    """
    Particle filter implementation for the Heston model, using modified parameters.

    Parameters:
    x (np.ndarray): Transformed parameters.
    simul (dict): Dictionary containing simulated noises.
    obs (np.ndarray): Observations, either 1D (log-returns) or 2D (log-returns and realized variance).
    transform_params (bool): If True, transform parameters to their original values
    c_i (int): Flag to determine if the filtered variance is computed.

    Returns:
    Tuple[float, np.ndarray]: Negative log likelihood and filtered average variance.
    """
    # Check if obs is 1D or 2D
    is_1D = obs.ndim == 1
    N = len(obs)

    # Retrieve random variables from simulated data
    U = simul['U']
    ZZ = simul['ZZ']
    UU = simul['UU']

    # Transform parameters to original scale
    if transform_params:
        mu = x[0] / 10
        kappa = np.exp(x[1])
        theta = np.exp(x[2])
        sigma = np.exp(x[3])
        rho = (1 - np.exp(-x[4])) / (1 + np.exp(-x[4]))
        eta_p = np.exp(x[5]) if not is_1D else None
    else: 
        mu = x[0] 
        kappa = x[1] 
        theta = x[2] 
        sigma = x[3] 
        rho = x[4] 
        eta_p = x[5] if not is_1D else None

    # Time step for one trading day
    delta = 1 / 252

    # Initial sampling of the volatility component
    v1 = U[:, 0] * 0.02

    # Pre-allocate vectors for filtered variance and log-likelihood
    V = np.zeros((N, 3)) if c_i == 1 else np.zeros(N)
    Like_H = np.zeros(N)

    # Define the probability density function
    def pdf(Y, RV, IV_s, v1, v2):
        if not is_1D: 
            return norm.pdf(Y, 
                            mu * delta - 0.5 * IV_s + rho / sigma *
                            (v2 - v1 - kappa * theta * delta + kappa * IV_s),
                            np.sqrt((1 - rho ** 2) * IV_s)) * (norm.pdf((IV_s - RV) / RV, 0, eta_p))
        else: 
            return norm.pdf(Y, 
                            mu * delta - 0.5 * IV_s + rho / sigma *
                            (v2 - v1 - kappa * theta * delta + kappa * IV_s),
                            np.sqrt((1 - rho ** 2) * IV_s)) 

    # apply the filtering 
    for t in tqdm(range(N), desc="filtering...", disable = not verbose):
        # Simulate one step propagation for volatility and compute integrated variance
        IV = 0
        vol_past = v1
        for d in range(5):
            v2 = np.maximum(vol_past + kappa * (theta - vol_past) * delta / 5 + sigma * np.sqrt(vol_past * delta / 5) * ZZ[:, t, d], 0.000001)
            IV += (vol_past + v2) / 2 * delta / 5
            vol_past = v2

        # Calculate weights for each particle and update log-likelihood
        Y = obs[t] if is_1D else obs[t, 0]
        RV = None if is_1D else obs[t, 1]
        w = pdf(Y, RV, IV, v1, v2)
        w_bar = np.mean(w) if np.mean(w) != 0 else 1e-6
        Like_H[t] = np.log(w_bar)

        # Normalize particle weights
        pis = w / w_bar

        # Apply smoothing to the volatility CDF
        arr = np.vstack((v2, pis)).T
        arr = arr[arr[:, 1] > 1e-7]
        arr = np.sort(arr, axis=0)
        arr[:, 1] = np.cumsum(arr[:, 1]) - arr[:, 1] / 2
        arr = np.vstack(([arr[0, 0] * 0.95, 0], arr, [arr[-1, 0] * 1.05, 1]))

        # Compute expected volatility value and confidence interval if required
        if c_i == 1:
            V[t, 0] = np.sum(v2 * pis)
            V[t, 1] = np.interp(0.025, arr[:, 1], arr[:, 0])
            V[t, 2] = np.interp(0.975, arr[:, 1], arr[:, 0])

        # Resample particles using smoothed CDF
        v1 = np.interp(UU[:, t], arr[:, 1], arr[:, 0])

    # Sum of negative log-likelihood
    Like_H = np.sum(-Like_H)
    
    # display likelihood if propmpted 
    if verbose: print("-log_likelihood: ", Like_H)

    return Like_H, V


