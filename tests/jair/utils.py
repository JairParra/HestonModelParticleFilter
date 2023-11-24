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
    
    plt.figure(figsize=(17, 7))

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
