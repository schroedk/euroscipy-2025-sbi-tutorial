"""
Lotka-Volterra Predator-Prey Model for SBI Tutorial
====================================================

This simulator models the population dynamics of wolves (predators)
and deer (prey) for environmental monitoring.

The model uses the classic Lotka-Volterra differential equations:
- dDeer/dt = α * Deer - β * Deer * Wolves
- dWolves/dt = δ * Deer * Wolves - γ * Wolves

Parameters:
- α: Deer birth rate
- β: Predation rate
- δ: Wolf efficiency converting deer to wolves
- γ: Wolf death rate
"""

from typing import Union

import numpy as np
import torch
from sbi.utils import BoxUniform
from scipy import stats


def lotka_volterra(
    y: np.ndarray, alpha: float, beta: float, delta: float, gamma: float
) -> np.ndarray:
    """Lotka-Volterra differential equations for deer-wolf dynamics."""
    deer, wolves = y
    ddeer_dt = alpha * deer - beta * deer * wolves
    dwolves_dt = delta * deer * wolves - gamma * wolves
    return np.asarray([ddeer_dt, dwolves_dt])


def simulate(parameters: np.ndarray, time_span: float = 200.0) -> np.ndarray:
    """Simulate deer-wolf population dynamics.

    Args:
        parameters: Array of [alpha, beta, delta, gamma] parameters
        time_span: Total simulation time in days (default: 200.0)

    Returns:
        Array of shape (timesteps, 2) with [deer, wolves] populations over time
    """
    alpha, beta, delta, gamma = parameters

    initial_populations = np.asarray([40.0, 9.0])  # [deer, wolves]
    dt = 0.1  # Time step

    timesteps = int(time_span / dt)
    populations = np.zeros((timesteps, 2))
    populations[0] = initial_populations

    for i in range(1, timesteps):
        populations[i] = (
            populations[i - 1]
            + lotka_volterra(populations[i - 1], alpha, beta, delta, gamma) * dt
        )

    return populations


def summarize_simulation(
    simulation_result: np.ndarray, use_autocorrelation: bool = False
) -> np.ndarray:
    """
    Convert simulation to summary statistics with observation noise.

    Calculates stats for each population (deer and wolves):
    - 5 moments: mean, std, max, skewness, kurtosis (always included)
    - 5 autocorrelation lags (optional, controlled by use_autocorrelation)
    """
    # Add observation noise to simulate real-world measurement uncertainty
    noise = np.random.randn(*simulation_result.shape)
    noisy_populations = simulation_result + noise

    deer_pop = noisy_populations[:, 0]
    wolves_pop = noisy_populations[:, 1]

    # --- Helper function to compute stats for one population ---
    def get_stats(population: np.ndarray) -> np.ndarray:
        # 5 moments
        moments = np.array(
            [
                np.mean(population),
                np.std(population),
                np.max(population),
                stats.skew(population),
                stats.kurtosis(population),
            ]
        )

        # 5 normalized autocorrelation lags at specific, spaced-out intervals
        mean_centered_pop = population - np.mean(population)
        autocorr_full = np.correlate(mean_centered_pop, mean_centered_pop, mode="full")

        # The value at lag 0 is the variance of the series.
        lag_0_corr = autocorr_full[autocorr_full.size // 2]

        # Avoid division by zero for constant series.
        if lag_0_corr > 1e-6:
            # Get the second half, normalize by lag 0.
            normalized_autocorr = (autocorr_full / lag_0_corr)[
                autocorr_full.size // 2 :
            ]

            # Take specific, spaced-out lags to capture longer-term dynamics.
            # These correspond to time delays of 1, 5, 10, 20, and 40 days.
            lags_to_take = [10, 50, 100, 200, 400]
            autocorr = normalized_autocorr[lags_to_take]
        else:
            # If variance is zero, autocorrelation is undefined, return zeros.
            autocorr = np.zeros(5)

        return (
            moments if not use_autocorrelation else np.concatenate([moments, autocorr])
        )

    # --- Calculate and combine stats for both populations ---
    deer_stats = get_stats(deer_pop)
    wolf_stats = get_stats(wolves_pop)

    summary = np.concatenate([deer_stats, wolf_stats])
    return summary


def lotka_volterra_simulator(
    params: Union[torch.Tensor, np.ndarray], use_autocorrelation: bool = False
) -> torch.Tensor:
    """SBI-compatible simulator that returns summary statistics."""
    # Convert parameters to numpy array
    if isinstance(params, torch.Tensor):
        params_np = params.detach().cpu().numpy()
    else:
        params_np = np.array(params)

    # Ensure parameters are positive (ecological constraint)
    params_np = np.abs(params_np)

    try:
        # Run simulation and get summary statistics
        simulation_result = simulate(params_np)
        summary_stats = summarize_simulation(
            simulation_result, use_autocorrelation=use_autocorrelation
        )
        return torch.tensor(summary_stats, dtype=torch.float32)
    except Exception as e:
        raise RuntimeError(f"Simulation failed with parameters {params_np}: {e}")


def get_summary_labels(use_autocorrelation: bool = False) -> list:
    """
    Get labels for summary statistics based on configuration.

    Args:
        use_autocorrelation: Whether autocorrelation features are included

    Returns:
        List of labels for all summary statistics
    """
    moment_labels = ["Mean", "Std", "Max", "Skew", "Kurtosis"]
    if use_autocorrelation:
        lags_taken = [10, 50, 100, 200, 400]
        acf_labels = [f"ACF Lag {lag}" for lag in lags_taken]
        stat_labels_per_pop = moment_labels + acf_labels
    else:
        stat_labels_per_pop = moment_labels

    all_labels = [f"Deer {label}" for label in stat_labels_per_pop] + [
        f"Wolf {label}" for label in stat_labels_per_pop
    ]

    return all_labels


def create_lotka_volterra_prior() -> BoxUniform:
    """Create a prior distribution for the Lotka-Volterra parameters."""

    lower_bound = torch.as_tensor([0.05, 0.01, 0.005, 0.005])
    upper_bound = torch.as_tensor([0.15, 0.03, 0.03, 0.15])
    prior = BoxUniform(low=lower_bound, high=upper_bound)

    return prior
