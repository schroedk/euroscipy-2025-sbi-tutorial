"""
Utility functions for SBI tutorial visualization and analysis.

This module contains helper functions for plotting posterior predictive
distributions and other visualizations used in the tutorial.
"""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import torch

if TYPE_CHECKING:
    from sbi.inference.posteriors.direct_posterior import DirectPosterior


def plot_posterior_predictions(
    posterior: "DirectPosterior",
    observed_data: torch.Tensor,
    simulate_func: callable,
    n_predictions: int = 1000,
    time_span: float = 200.0,
    dt: float = 0.1,
    figsize: tuple = (14, 8),
) -> None:
    """
    Plot posterior predictive time series with uncertainty bands.

    Args:
        posterior: Trained SBI posterior distribution
        observed_data: Observed summary statistics used for conditioning
        simulate_func: Function to simulate time series from parameters
        n_predictions: Number of posterior samples to use for predictions
        time_span: Total simulation time
        dt: Time step for simulation
        figsize: Figure size for the plot
    """
    print("🔮 Generating future time series predictions...")

    # Sample parameters from the posterior
    param_samples = posterior.sample((n_predictions,), x=observed_data)

    # Simulate the full time series for each parameter set
    all_predictions = []
    for params in param_samples:
        # Ensure parameters are positive
        params_np = params.detach().cpu().numpy()
        time_series = simulate_func(params_np)
        all_predictions.append(time_series)

    # Stack predictions into a single tensor for analysis
    # Shape: (n_predictions, timesteps, 2)
    predictions_tensor = torch.tensor(np.array(all_predictions), dtype=torch.float32)

    # Calculate statistics over time
    # Get the Maximum a Posteriori (MAP) estimate to plot a single "best" trajectory
    posterior.set_default_x(observed_data)
    map_estimate = posterior.map().squeeze()
    map_prediction = torch.tensor(simulate_func(map_estimate.numpy()))

    # Calculate percentiles for uncertainty bands
    lower_bound = torch.quantile(predictions_tensor, 0.05, dim=0)
    upper_bound = torch.quantile(predictions_tensor, 0.95, dim=0)
    median_prediction = torch.quantile(predictions_tensor, 0.5, dim=0)

    # Create the time axis for plotting
    time_axis = np.arange(0, time_span, dt)

    # Visualize the time series predictions
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the MAP prediction
    ax.plot(
        time_axis,
        map_prediction[:, 0],
        color="darkred",
        linestyle="--",
        lw=2,
        label="MAP Deer Prediction",
    )
    ax.plot(
        time_axis,
        map_prediction[:, 1],
        color="darkblue",
        linestyle="--",
        lw=2,
        label="MAP Wolf Prediction",
    )

    # Plot the median and uncertainty bands for deer
    ax.plot(
        time_axis,
        median_prediction[:, 0],
        color="red",
        lw=2,
        label="Median Deer Prediction",
    )
    ax.fill_between(
        time_axis,
        lower_bound[:, 0],
        upper_bound[:, 0],
        color="red",
        alpha=0.2,
        label="90% Credible Interval (Deer)",
    )

    # Plot the median and uncertainty bands for wolves
    ax.plot(
        time_axis,
        median_prediction[:, 1],
        color="blue",
        lw=2,
        label="Median Wolf Prediction",
    )
    ax.fill_between(
        time_axis,
        lower_bound[:, 1],
        upper_bound[:, 1],
        color="blue",
        alpha=0.2,
        label="90% Credible Interval (Wolves)",
    )

    ax.set_xlabel("Time (days)", fontsize=14)
    ax.set_ylabel("Population", fontsize=14)
    ax.set_title("Predicted Future Population Dynamics", fontsize=18)
    ax.legend(loc="upper right")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()

    print("\n✅ Successfully generated and plotted future population dynamics.")
    print(
        "💡 Notice how the uncertainty grows over time, a hallmark of realistic forecasting."
    )


def print_summary_statistics(
    observed_data: torch.Tensor, labels: list, use_autocorrelation: bool
) -> None:
    """
    Print observed summary statistics in a formatted way.

    Args:
        observed_data: Tensor of observed summary statistics
        labels: List of labels for each statistic
        use_autocorrelation: Whether autocorrelation stats are included
    """
    print("🐺🦌 Observed Summary Statistics from Forests South of Kraków:")
    print("=" * 60)

    # Print the observed data in a more readable format
    for label, value in zip(labels, observed_data):
        print(f"{label:18s}: {value:.2f}")

    print(f"\n📈 Total summary statistics: {len(observed_data)}")
    print(
        "\n❓ Question: What birth and death rates could produce these population patterns?"
    )
    if use_autocorrelation:
        print(
            "💡 We have rich temporal information - this should enable good long-term predictions!"
        )
    else:
        print("💡 We only have basic moments - let's see how far this gets us.")
