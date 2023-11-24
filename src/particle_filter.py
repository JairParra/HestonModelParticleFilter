"""
simulate_heston.py 

This scripts simulates Heston's 93 model according to 
"""

##################
### 1. Imports ###
##################

import warnings 
import numpy as np 
from scipy.stats import norm
from typing import Tuple, Optional

##################
### 2. Helpers ###
##################

##########################
### 3. Particle Filter ###
##########################

def heston93_particle_filter(x: np.ndarray, simul: dict, obs: np.ndarray, c_i: int) -> Tuple[float, np.ndarray]:
    """
    Particle filter implementation for the Heston model.

    Parameters:
    x (np.ndarray): Transformed parameters.
    simul (dict): Dictionary containing simulated noises.
    obs (np.ndarray): Observations, either 1D (log-returns) or 2D (log-returns and realized variance).
    c_i (int): Flag to determine if the filtered variance is computed.

    Returns:
    Tuple[float, np.ndarray]: Negative log likelihood and filtered average variance.
    """
    # Determine the number of observations based on the input
    size_obs = obs.shape[1]
    N = len(obs)

    # Retrieve uniform and normal random variables from simulated data
    U = simul['U']
    ZZ = simul['ZZ']
    UU = simul['UU']

    # Separate log-returns and realized variance if available
    if size_obs == 2:
        Y, RV = obs[:, 0], obs[:, 1]
        eta_p = np.exp(x[5])
    else:
        Y = obs[:, 0]

    # Transform parameters back to original scale
    mu = x[0] / 10
    kappa = np.exp(x[1])
    theta = np.exp(x[2])
    sigma = np.exp(x[3])
    rho = (1 - np.exp(-x[4])) / (1 + np.exp(-x[4]))

    # Set the time step for one trading day
    delta = 1 / 252

    # Initial sampling of the volatility component
    v1 = U[:, 0] * 0.02

    # Pre-allocate vectors for filtered variance and log-likelihood
    V = np.zeros((N, 3)) if c_i == 1 else np.zeros(N)
    Like_H = np.zeros(N)

    # Define the probability density function based on the observation set
    pdf = lambda Y, RV, IV_s, v1, v2: norm.pdf(Y,
                                               mu * delta - 0.5 * IV_s + rho / sigma *
                                               (v2 - v1 - kappa * theta * delta + kappa * IV_s),
                                               np.sqrt((1 - rho ** 2) * IV_s)) * (norm.pdf((IV_s - RV) / RV, 0, eta_p) 
                                                                                  if size_obs == 2 else 1)

    for t in range(N):
        # Simulate one step propagation for volatility and compute integrated variance
        IV = 0
        vol_past = v1
        for d in range(5):
            v2 = np.maximum(vol_past + kappa * (theta - vol_past) * delta / 5 + sigma *
                            np.sqrt(vol_past * delta / 5) * ZZ[:, t, d], 0.000001)
            IV += (vol_past + v2) / 2 * delta / 5
            vol_past = v2

        # Calculate weights for each particle
        w = pdf(Y[t], RV[t] if size_obs == 2 else None, IV, v1, v2)
        
        # Compute the Log-Likelihood            
        Like_H[t] = np.log(np.mean(w))

        # Normalize particle weights
        pis = w / np.sum(w)

        # Apply Malik & Pitt smoothing to stabilize the volatility CDF
        arr = np.vstack((v2, pis)).T
        arr = arr[pis > 1e-7]
        arr = np.sort(arr, axis=0)
        arr[:, 1] = np.cumsum(arr[:, 1]) - arr[:, 1] / 2
        arr = np.vstack(([arr[0, 0] * 0.95, 0], arr, [arr[-1, 0] * 1.05, 1]))

        # Compute expected volatility value and confidence interval if required
        if c_i == 1:
            V[t, 0] = np.sum(v2 * pis)  # expected volatility
            V[t, 1] = np.interp(0.025, arr[:, 1], arr[:, 0])
            V[t, 2] = np.interp(0.975, arr[:, 1], arr[:, 0])

        # Resample particles using smoothed CDF
        v1 = np.interp(UU[:, t], arr[:, 1], arr[:, 0])

    # Sum of negative log-likelihood
    Like_H = np.sum(-Like_H)

    return Like_H, V



