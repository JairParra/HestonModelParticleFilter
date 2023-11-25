"""
plotting.py 

Contains scripts for all plots employed in the simulation. 

@author: Hair Albeiro Parra Barrera
"""
################################################################################
##################
### 1. Imports ###
##################

# general imports
import pprint 
import seaborn as sns
import matplotlib.pyplot as plt

## configurations 

# setup pretty printing
pp = pprint.PrettyPrinter(indent = 4)

# setup dark plotting
sns.set(style="darkgrid", context="talk")
plt.style.use("dark_background")
plt.rcParams.update({"grid.linewidth":0.5, "grid.alpha":0.5})

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
    
    
########################################
### 3. Plots for volatility analysis ###
########################################
    
# Function for Plots Using Only Returns
def plot_returns_analysis(ret, V, V_ret_true, V_ret_est, num_days):
    plt.figure(figsize=(10, 12))
    plt.suptitle('Volatility Analysis with Returns')

    # Plot of simulated returns
    plt.subplot(4, 1, 1)
    plt.plot(ret, 'k')
    plt.xlabel('Days')
    plt.ylabel('Log-Returns')
    plt.title('Simulated Returns')
    plt.xlim([0, num_days])

    # Volatility filtered using returns and true parameters
    plt.subplot(4, 1, 2)
    plt.plot(V, 'k', label='True Variance')
    plt.plot(V_ret_true[:, 0], 'b', label='Filtered Variance')
    plt.plot(V_ret_true[:, 1], 'r--', label='2.5%')
    plt.plot(V_ret_true[:, 2], 'r--', label='97.5%')
    plt.title('Volatility Filtered (Returns & True Params)')
    plt.xlabel('Days')
    plt.ylabel('Volatility Level')
    plt.legend(loc='upper left') 
    plt.xlim([0, num_days])

    # Volatility filtered using returns and estimated parameters
    plt.subplot(4, 1, 3)
    plt.plot(V, 'k', label='True Variance')
    plt.plot(V_ret_est[:, 0], 'b', label='Filtered Variance')
    plt.plot(V_ret_est[:, 1], 'r--', label='2.5%')
    plt.plot(V_ret_est[:, 2], 'r--', label='97.5%')
    plt.title('Volatility Filtered (Returns & Est. Params)')
    plt.xlabel('Days')
    plt.ylabel('Volatility Level')
    plt.legend(loc='upper left')
    plt.xlim([0, num_days])

    # Comparison of filtered volatility (true vs estimated parameters)
    plt.subplot(4, 1, 4)
    plt.plot(V, 'k', label='True Variance')
    plt.plot(V_ret_true[:, 0], 'r', label='Filtered (True Params)')
    plt.plot(V_ret_est[:, 0], 'b', label='Filtered (Est. Params)')
    plt.title('Comparison of Filtered Volatility')
    plt.xlabel('Days')
    plt.ylabel('Volatility Level')
    plt.legend(loc='upper left')
    plt.xlim([0, num_days])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# Function for Plots Using Returns and Realized Variance
def plot_returns_rv_analysis(ret, V, V_RV_true, V_RV_est, num_days):
    plt.figure(figsize=(10, 12))
    plt.suptitle('Volatility Analysis with Returns & RV')

    # Plot of simulated returns
    plt.subplot(4, 1, 1)
    plt.plot(ret, 'k')
    plt.xlabel('Days')
    plt.ylabel('Log-Returns')
    plt.title('Simulated Returns')
    plt.xlim([0, num_days])

    # Volatility filtered using returns + RV and the true parameters
    plt.subplot(4, 1, 2)
    plt.plot(V, 'k', label='True Variance')
    plt.plot(V_RV_true[:, 0], 'b', label='Filtered Variance')
    plt.plot(V_RV_true[:, 1], 'r--', label='2.5%')
    plt.plot(V_RV_true[:, 2], 'r--', label='97.5%')
    plt.title('Volatility Filtered (Returns + RV & True Params)')
    plt.xlabel('Days')
    plt.ylabel('Volatility Level')
    plt.legend(loc='upper left')
    plt.xlim([0, num_days])

    # Volatility filtered using returns + RV and the estimated parameters
    plt.subplot(4, 1, 3)
    plt.plot(V, 'k', label='True Variance')
    plt.plot(V_RV_est[:, 0], 'b', label='Filtered Variance')
    plt.plot(V_RV_est[:, 1], 'r--', label='2.5%')
    plt.plot(V_RV_est[:, 2], 'r--', label='97.5%')
    plt.title('Volatility Filtered (Returns + RV & Est. Params)')
    plt.xlabel('Days')
    plt.ylabel('Volatility Level')
    plt.legend(loc='upper left')
    plt.xlim([0, num_days])

    # Comparison of filtered volatility (true vs estimated parameters)
    plt.subplot(4, 1, 4)
    plt.plot(V, 'k', label='True Variance')
    plt.plot(V_RV_true[:, 0], 'r', label='Filtered (True Params)')
    plt.plot(V_RV_est[:, 0], 'b', label='Filtered (Est. Params)')
    plt.title('Comparison of Filtered Volatility')
    plt.xlabel('Days')
    plt.ylabel('Volatility Level')
    plt.legend(loc='upper left')
    plt.xlim([0, num_days])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

########################################
### 4. Plots for likelihood analysis ###
########################################

def plot_log_likelihood_ret(log_likelihoods, num_days):
    """
    Plot log-likelihood function shapes for different parameters using returns only.
    
    Args:
    log_likelihoods (dict): Dictionary of log likelihood values for each parameter.
    num_days (int): Number of days in the simulation.
    """
    plt.figure(figsize=(12, 8))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    for i, (param_name, data) in enumerate(log_likelihoods.items(), 1):
        plt.subplot(3, 2, i)
        values, likelihoods = data[0], data[1]  # Correctly access the tuple elements
        plt.plot(values, likelihoods, 'k')
        plt.ylabel('Log-Likelihood')
        plt.xlabel(param_name)
        plt.title(f'Parameter {param_name}')
    
    plt.tight_layout()
    plt.show()


def plot_log_likelihood_ret_rv(log_likelihoods, num_days):
    """
    Plot log-likelihood function shapes for different parameters using returns and RV.
    
    Args:
    log_likelihoods (dict): Dictionary of log likelihood values for each parameter.
    num_days (int): Number of days in the simulation.
    """
    plt.figure(figsize=(12, 8))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    for i, (param_name, data) in enumerate(log_likelihoods.items(), 1):
        plt.subplot(3, 2, i)
        values, likelihoods = data[0], data[1]  # Correctly access the tuple elements
        plt.plot(values, likelihoods, 'k')
        plt.ylabel('Log-Likelihood')
        plt.xlabel(param_name)
        plt.title(f'Parameter {param_name}')
    plt.tight_layout()
    plt.show()

