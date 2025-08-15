"""
SIR Epidemic Model Simulator for SBI Tutorial
=============================================

This simulator models disease spread through a population using the
classic Susceptible-Infected-Recovered (SIR) compartmental model.

The model describes how individuals move between compartments:
- Susceptible (S): Can catch the disease
- Infected (I): Currently sick and contagious  
- Recovered (R): Immune after recovery

This is suitable for demonstrating SBI on an epidemiological problem.
"""

import torch
import numpy as np
from typing import Union, Tuple


def sir_epidemic_simulator(
    params: Union[torch.Tensor, np.ndarray],
    population_size: int = 10000,
    return_time_series: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
    """
    Simulate an SIR (Susceptible-Infected-Recovered) epidemic model.
    
    The SIR model describes disease spread through a population:
    dS/dt = -β·S·I/N  (susceptible → infected)
    dI/dt = β·S·I/N - γ·I  (infected → recovered)
    dR/dt = γ·I  (recovered)
    
    Parameters
    ----------
    params : torch.Tensor or np.ndarray with shape (3,)
        [0] beta: Infection rate (contacts per day × transmission probability)
              Range [0.1, 2.0] - higher means more contagious
        [1] gamma: Recovery rate (1/infectious period in days)
              Range [0.05, 0.5] - higher means faster recovery
        [2] I0: Initial number of infected individuals
              Range [1, 100]
    
    population_size : int, optional
        Total population size (default: 10,000)
    
    return_time_series : bool, optional
        If True, also returns full time series for visualization
    
    Returns
    -------
    observations : torch.Tensor with shape (4,)
        [0] peak_infected: Maximum number of infected at any time
        [1] time_to_peak: Days until peak infection
        [2] total_recovered: Final number of recovered individuals
        [3] epidemic_duration: Days until infection drops below 1
    
    If return_time_series=True, also returns dict with:
        't': time points
        'S': susceptible over time
        'I': infected over time
        'R': recovered over time
    
    Examples
    --------
    >>> params = torch.tensor([0.5, 0.1, 10])  # R0 ≈ 5
    >>> observations = sir_epidemic_simulator(params)
    >>> print(f"Peak: {observations[0]:.0f} infected on day {observations[1]:.0f}")
    """
    # Convert to numpy
    if isinstance(params, torch.Tensor):
        params_np = params.detach().cpu().numpy()
    else:
        params_np = np.array(params)
    
    beta = params_np[0]  # Infection rate
    gamma = params_np[1]  # Recovery rate
    I0 = int(params_np[2])  # Initial infected
    
    # Initial conditions (fractions of population)
    N = float(population_size)
    S = (N - I0) / N
    I = I0 / N
    R = 0.0
    
    # Time settings
    dt = 0.1  # Days
    t = 0.0
    max_time = 365  # Maximum simulation time (1 year)
    
    # Storage
    if return_time_series:
        t_series = [t]
        S_series = [S * N]
        I_series = [I * N]
        R_series = [R * N]
    
    # Track observables
    peak_infected = I * N
    time_to_peak = 0.0
    epidemic_duration = 0.0
    
    # Simulate epidemic
    while t < max_time and I > 1e-6:  # Stop when effectively no infections
        # SIR differential equations
        dS = -beta * S * I * dt
        dI = (beta * S * I - gamma * I) * dt
        dR = gamma * I * dt
        
        # Update state
        S = max(0, S + dS)
        I = max(0, I + dI)
        R = min(1, R + dR)
        
        t += dt
        
        # Track peak
        current_infected = I * N
        if current_infected > peak_infected:
            peak_infected = current_infected
            time_to_peak = t
        
        # Track duration (last time above 1 infected)
        if current_infected >= 1:
            epidemic_duration = t
        
        if return_time_series:
            t_series.append(t)
            S_series.append(S * N)
            I_series.append(I * N)
            R_series.append(R * N)
    
    # Final statistics
    total_recovered = R * N
    
    # Add observation noise (5% relative error for counts, 10% for times)
    peak_infected *= (1 + np.random.randn() * 0.05)
    time_to_peak *= (1 + np.random.randn() * 0.10)
    total_recovered *= (1 + np.random.randn() * 0.05)
    epidemic_duration *= (1 + np.random.randn() * 0.10)
    
    # Ensure sensible values
    peak_infected = np.clip(peak_infected, 1, N)
    time_to_peak = max(1, time_to_peak)
    total_recovered = np.clip(total_recovered, 0, N)
    epidemic_duration = max(time_to_peak, epidemic_duration)
    
    observations = torch.tensor(
        [peak_infected, time_to_peak, total_recovered, epidemic_duration],
        dtype=torch.float32
    )
    
    if return_time_series:
        time_series = {
            't': np.array(t_series),
            'S': np.array(S_series),
            'I': np.array(I_series),
            'R': np.array(R_series)
        }
        return observations, time_series
    
    return observations


def create_sir_prior():
    """
    Create prior distribution for SIR model parameters.
    
    Returns
    -------
    prior : sbi.utils.BoxUniform
        Prior distribution over [beta, gamma, I0]
    """
    from sbi import utils
    
    # Reasonable ranges for epidemic parameters
    low = torch.tensor([0.1, 0.05, 1.0])   # [beta, gamma, I0]
    high = torch.tensor([2.0, 0.5, 100.0])
    
    return utils.BoxUniform(low=low, high=high)


def calculate_R0(beta: float, gamma: float) -> float:
    """
    Calculate the basic reproduction number R0.
    
    R0 represents the average number of secondary infections
    caused by one infected individual in a fully susceptible population.
    
    Parameters
    ----------
    beta : float
        Infection rate
    gamma : float
        Recovery rate
        
    Returns
    -------
    R0 : float
        Basic reproduction number
        
    Notes
    -----
    - R0 < 1: Epidemic dies out
    - R0 = 1: Endemic equilibrium
    - R0 > 1: Epidemic spreads
    """
    return beta / gamma


def generate_observation(true_params: Union[None, torch.Tensor] = None, seed: int = 42):
    """
    Generate synthetic observed data for the tutorial.
    
    Args:
        true_params: True parameters (if None, uses default)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (observations, true_parameters)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if true_params is None:
        # Default parameters: moderate outbreak (R0 ≈ 3)
        true_params = torch.tensor([0.3, 0.1, 10])
    
    observations = sir_epidemic_simulator(true_params)
    return observations, true_params


# Quick test
if __name__ == "__main__":
    print("Testing SIR Epidemic Simulator...")
    
    # Test epidemic
    params = torch.tensor([0.5, 0.1, 10])  # β=0.5, γ=0.1, I0=10
    obs = sir_epidemic_simulator(params)
    
    R0 = calculate_R0(params[0].item(), params[1].item())
    print(f"Input: β={params[0]:.2f}, γ={params[1]:.2f}, I₀={params[2]:.0f}")
    print(f"Basic reproduction number: R₀={R0:.1f}")
    print(f"\nOutput:")
    print(f"  Peak infected: {obs[0]:.0f} people")
    print(f"  Time to peak: {obs[1]:.0f} days")
    print(f"  Total recovered: {obs[2]:.0f} people")
    print(f"  Epidemic duration: {obs[3]:.0f} days")
    
    # Test with time series
    obs, ts = sir_epidemic_simulator(params, return_time_series=True)
    print(f"\nTime series: {len(ts['t'])} time points")
    
    # Test prior
    prior = create_sir_prior()
    print(f"Prior created: {prior}")
    
    print("\n✅ SIR simulator ready for tutorial!")
