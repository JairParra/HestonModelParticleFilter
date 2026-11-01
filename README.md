# Heston Model Particle Filter

## Report

**Overleaf Edit Link:** <https://www.overleaf.com/6361181294brxzmrrqfxbg#e93e60>

Our report describing the theoretical and technical aspects can be found here: **TODO: UPDATE LINK**

## Heston Model

Heston (1993) proposed a stochastic volatility model where the asset price $S$ follows:

$$
\begin{aligned}
dS_t &= \mu S_t dt + \sqrt{V_t} S_t dW_t^P \\
dV_t &= \kappa (\theta - V_t) dt + \sigma \sqrt{V_t} dB_t^P
\end{aligned}
$$

$W^P$ and $B^P$ are two correlated Brownian motions, and $\rho$ denotes their correlation. Using Itô's lemma, the asset log-price satisfies:

$$
\begin{aligned}
d \ln S_t &= \left( \mu - \frac{V_t}{2} \right) dt + \sqrt{V_t} dW_t^P \\
dV_t &= \kappa (\theta - V_t) dt + \sigma \sqrt{V_t} dB_t^P
\end{aligned}
$$

The model estimation requires a discrete-time version. We use the convention that capital letters represent the continuous-time model where time $t$ is expressed in years, and lowercase notation is for the discrete-time framework in which time $t$ is expressed in number of periods.

Let $y_t = \log S_t - \log S_{t-1}$ be the log-price variation (log-return), and $v_t = \sqrt{V_t}$:

$$
y_{t+1} = \mu - \frac{1}{2} v_t^2 + v_t \sqrt{\Delta} z_t^W
$$

$$
v_{t+1} = v_t + \kappa(\theta - v_t \Delta) - \kappa \int_{t}^{t+\Delta} \sqrt{V_s} ds + \sigma \sqrt{\Delta} z_t^B
$$

For a daily discretization, the time step is $\Delta = 1/252$. The observed sample of returns up to time $t$ is $y_{1:t} = (y_1, y_2, \dots, y_t)$.

To use this model, we have to:
1. Estimate the latent (unobservable) process $\{v_t\}_{t=1}^T$
2. Estimate the model's parameters $\Theta = \{\mu, \kappa, \theta, \sigma, \rho\}$

We use the Sequential Monte Carlo (SMC) algorithm, also known as Particle Filtering, developed by Gordon et al. (1993).
