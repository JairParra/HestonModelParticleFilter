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

## configurations 

# setup pretty printing
pp = pprint.PrettyPrinter(indent = 4)

# setup dark plotting
sns.set(style="darkgrid", context="talk")
plt.style.use("dark_background")
plt.rcParams.update({"grid.linewidth":0.5, "grid.alpha":0.5})

################################################################################
#########################
### 2. Plotting Utils ###
#########################

def plot_simulations(ret, V, IV, num_days):
    days = range(1, num_days + 1)
    
    plt.figure(figsize=(18, 6))

    # Plot returns
    plt.subplot(3, 1, 1)  # 3 rows, 1 column, 1st subplot
    plt.plot(days, ret, label='Returns')
    plt.title('Simulated Daily Returns')
    plt.xlabel('Day')
    plt.ylabel('Returns')
    plt.legend()

    # Plot volatility
    plt.subplot(3, 1, 2)  # 3 rows, 1 column, 2nd subplot
    plt.plot(days, V, label='Volatility', color='orange')
    plt.title('Simulated Daily Volatility')
    plt.xlabel('Day')
    plt.ylabel('Volatility')
    plt.legend()

    # Plot integrated variance
    plt.subplot(3, 1, 3)  # 3 rows, 1 column, 3rd subplot
    plt.plot(days, IV, label='Integrated Variance', color='green')
    plt.title('Simulated Integrated Variance')
    plt.xlabel('Day')
    plt.ylabel('Integrated Variance')
    plt.legend()

    plt.tight_layout()
    plt.show()



def plot_filtered_variance(filtered_variance, num_days, include_confidence_interval=True):
    """
    Plots the progression of filtered variance over time.

    Parameters:
    filtered_variance (np.ndarray): Filtered variance array from the particle filter.
    num_days (int): Number of days (time steps) in the simulation.
    include_confidence_interval (bool): Whether to include confidence intervals in the plot.
    """
    time_steps = range(num_days)
    
    # Extract filtered variance values
    var_values = filtered_variance[:, 0]

    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, var_values, label='Filtered Variance', color='blue')
    
    if include_confidence_interval and filtered_variance.shape[1] > 1:
        # Extract lower and upper bounds for the confidence interval
        lower_bound = filtered_variance[:, 1]
        upper_bound = filtered_variance[:, 2]
        
        plt.fill_between(time_steps, lower_bound, upper_bound, color='gray', alpha=0.3,
                         label='95% Confidence Interval')

    plt.xlabel('Time (Days)')
    plt.ylabel('Variance')
    plt.title('Filtered Variance Progression Over Time')
    plt.legend()
    plt.show()
    
    
