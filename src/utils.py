"""
Utility functions for SBI tutorial visualization and analysis.

This module contains helper functions for plotting posterior predictive
distributions and other visualizations used in the tutorial.
"""

import itertools
import math
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sbi.inference import NeuralInference


def plot_posterior_predictions(
    predictions: torch.Tensor,
    map_prediction: torch.Tensor | None = None,
    time_span: float = 200.0,
    dt: float = 0.1,
    figsize: tuple = (14, 8),
) -> tuple[Figure, Axes]:
    """
    Plot posterior predictive time series with uncertainty bands.

    Args:
        predictions: Tensor of shape (n_predictions, timesteps, 2) containing
                      all simulated time series
        map_prediction: Optional MAP prediction tensor of shape (timesteps, 2).
                       If None, MAP will not be plotted.
        time_span: Total simulation time
        dt: Time step for simulation
        figsize: Figure size for the plot
    """
    print("Plotting posterior predictive time series...")

    # Ensure predictions_tensor is the right shape
    assert (
        predictions.ndim == 3
    ), "predictions_tensor should have shape (n_predictions, timesteps, 2)"
    assert (
        predictions.shape[2] == 2
    ), "predictions_tensor should have 2 species in last dimension"

    # Calculate percentiles for uncertainty bands
    lower_bound = torch.quantile(predictions, 0.05, dim=0)
    upper_bound = torch.quantile(predictions, 0.95, dim=0)

    # Create the time axis for plotting
    time_axis = np.arange(0, time_span, dt)

    # Visualize the time series predictions
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the MAP prediction if provided
    if map_prediction is not None:
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

    # Plot uncertainty bands for deer
    ax.fill_between(
        time_axis,
        lower_bound[:, 0],
        upper_bound[:, 0],
        color="red",
        alpha=0.2,
        label="90% Credible Interval (Deer)",
    )

    # Plot uncertainty bands for wolves
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

    return fig, ax


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

    if observed_data.ndim > 1:
        observed_data = observed_data.squeeze()

    assert observed_data.ndim == 1, "Observed data should be a 1D tensor."
    assert len(observed_data) == len(
        labels
    ), "Mismatch between observed data and labels."

    # Print the observed data in a more readable format
    for label, value in zip(labels, observed_data, strict=False):
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
        print(
            "💡 We only have basic moments as summary statistics - let's see how far this gets us."
        )


def generate_posterior_predictive_simulations(
    posterior,
    observed_data: torch.Tensor,
    simulate_func: Callable,
    prior: torch.distributions.Distribution,
    num_simulations: int = 1000,
    num_workers: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate posterior predictive simulations and MAP simulation.

    Args:
        posterior: Trained SBI posterior distribution
        observed_data: Observed summary statistics used for conditioning
        simulate_func: Function to simulate time series from parameters
        prior: Prior distribution (needed for process_simulator)
        num_simulations: Number of posterior samples to simulate
        num_workers: Number of workers for parallel simulation

    Returns:
        map_simulation: MAP simulation tensor of shape (timesteps, 2)
        predictive_simulations: Tensor of shape (num_simulations, timesteps, 2)
    """
    from sbi.inference import simulate_for_sbi
    from sbi.utils.user_input_checks import process_simulator

    # Set the observed data as default for the posterior
    posterior.set_default_x(observed_data)

    # Process the simulator to work with batched inputs
    batch_simulator = process_simulator(simulate_func, prior, True)

    # Generate posterior samples and their corresponding simulations
    _, predictive_simulations = simulate_for_sbi(
        batch_simulator,
        posterior,
        num_simulations=num_simulations,
        num_workers=num_workers,
    )

    # Obtain the MAP estimate using gradient ascent on the posterior
    map_estimate = posterior.map()
    map_simulation = simulate_func(map_estimate.numpy().squeeze())

    # Convert to tensors
    map_simulation_tensor = torch.from_numpy(map_simulation)

    return map_simulation_tensor, predictive_simulations


def analyze_posterior_statistics(
    posterior_samples: torch.Tensor,
    param_names: list[str],
    true_params: torch.Tensor | list[float],
) -> dict[str, Any]:
    """
    Analyze and print posterior statistics including correlations.

    Args:
        posterior_samples: torch.Tensor of shape (n_samples, n_params)
        param_names: List of parameter names/descriptions
        true_params: True parameter values for comparison

    Returns:
        dict with statistics: means, stds, medians, ci_lower, ci_upper, correlations
    """
    # Convert true_params to tensor if it's a list
    if isinstance(true_params, list):
        true_params = torch.tensor(true_params)

    # Calculate posterior statistics
    posterior_mean: torch.Tensor = posterior_samples.mean(dim=0)
    posterior_std: torch.Tensor = posterior_samples.std(dim=0)
    posterior_median: torch.Tensor = posterior_samples.median(dim=0).values

    # Calculate 95% credible intervals
    lower_ci: torch.Tensor = torch.quantile(posterior_samples, 0.025, dim=0)
    upper_ci: torch.Tensor = torch.quantile(posterior_samples, 0.975, dim=0)

    # Calculate posterior correlations
    posterior_corr: np.ndarray = np.corrcoef(posterior_samples.detach().cpu().numpy().T)

    # Print statistics table
    print("📊 Posterior Statistics")
    print("=" * 70)
    print(f"{'Parameter':<20} {'Mean ± Std':<20} {'Median':<15} {'95% CI':<20}")
    print("-" * 70)

    for i, name in enumerate(param_names):
        mean_std = f"{posterior_mean[i]:.3f} ± {posterior_std[i]:.3f}"
        median = f"{posterior_median[i]:.3f}"
        ci = f"[{lower_ci[i]:.3f}, {upper_ci[i]:.3f}]"
        print(f"{name:<20} {mean_std:<20} {median:<15} {ci:<20}")

    # Print true parameters comparison
    print("\n🎯 True Parameters (for comparison):")
    for i, name in enumerate(param_names):
        in_ci = bool(lower_ci[i] <= true_params[i] <= upper_ci[i])
        symbol = "✅" if in_ci else "❌"
        print(f"{symbol} {name:<20} {true_params[i]:.3f} (in 95% CI: {in_ci})")

    # Print correlations
    print("\n🔗 Parameter Correlations:")
    print("=" * 50)

    # Generate all unique parameter pairs dynamically
    n_params = posterior_samples.shape[1]
    param_pairs = list(itertools.combinations(range(n_params), 2))

    for i, j in param_pairs:
        corr_value = posterior_corr[i, j]
        name_i = param_names[i].split("(")[0].strip()
        name_j = param_names[j].split("(")[0].strip()
        print(f"{name_i} vs {name_j:<12}: {corr_value:+.3f}")

    # Return statistics dictionary
    return {
        "means": posterior_mean,
        "stds": posterior_std,
        "medians": posterior_median,
        "ci_lower": lower_ci,
        "ci_upper": upper_ci,
        "correlations": posterior_corr,
    }


def _get_grid_layout(n_dims: int) -> tuple[int, int]:
    """
    Determine grid layout for plotting multiple dimensions using simple modulo operation.

    Args:
        n_dims: Number of dimensions to plot

    Returns:
        Tuple of (rows, cols) for subplot grid

    Raises:
        ValueError: If n_dims > 20
    """
    if n_dims > 20:
        raise ValueError(f"Too many dimensions ({n_dims}). Maximum supported is 20.")

    if n_dims == 1:
        return 1, 1

    # Use 5 columns for simplicity, calculate rows needed
    cols = 5
    rows = math.ceil(n_dims / cols)
    return rows, cols


def plot_predictive_check(
    x: torch.Tensor,
    observed_data: torch.Tensor,
    stat_names: list[str],
    title: str = "Predictive Check: Is our observed data consistent with the samples?",
    limits: list[tuple[float, float]] | None = None,
    percentile_allowance: float = 95.0,
) -> tuple[Figure, list[Axes]]:
    """
    Perform and visualize a predictive check (prior or posterior).

    This function checks whether observed data falls within the range of data
    that the samples can generate by plotting histograms of predictive samples
    and marking observed values.

    Args:
        x: Predictive samples of shape (n_samples, n_dims)
        observed_data: Observed summary statistics, shape (n_dims,) or (1, n_dims)
        stat_names: List of names for each statistic/dimension
        title: Title for the plot
        limits: Optional list of (lower, upper) limits for each statistic's x-axis.
               If None, uses data-driven limits. Useful for comparing different
               predictive checks (e.g., using prior limits for posterior check).
    """
    # Ensure observed_data is properly shaped
    if observed_data.ndim > 1:
        observed_data = observed_data.flatten()

    n_dims = x.shape[1]

    # Validate inputs
    assert (
        len(stat_names) == n_dims
    ), f"Length of stat_names ({len(stat_names)}) must match number of dimensions ({n_dims})"
    assert (
        len(observed_data) == n_dims
    ), f"Length of observed_data ({len(observed_data)}) must match number of dimensions ({n_dims})"

    if limits is not None:
        assert (
            len(limits) == n_dims
        ), f"Length of limits ({len(limits)}) must match number of dimensions ({n_dims})"

    # Get optimal grid layout
    rows, cols = _get_grid_layout(n_dims)

    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    # Handle single subplot case
    if n_dims == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Plot each dimension
    for i in range(n_dims):
        ax = axes[i]

        # Plot histogram of predictive samples
        ax.hist(x[:, i].numpy(), bins=30, alpha=0.7, color="blue", density=True)

        # Mark observed value
        obs_val = observed_data[i].item()
        ax.axvline(obs_val, color="red", linewidth=2, label="Observed")

        # Calculate percentile
        percentile = (x[:, i] < obs_val).float().mean() * 100

        ax.set_title(f"{stat_names[i]}\n(p={percentile:.1f}%)")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")

        # Set limits if provided
        if limits is not None:
            ax.set_xlim(limits[i])

        if i == 0:  # Add legend to first subplot
            ax.legend()

    # Hide unused subplots
    for i in range(n_dims, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

    # Assessment
    print("\n📊 Predictive Check Assessment:")
    print("-" * 40)
    percentiles = [
        (x[:, i] < observed_data[i]).float().mean() * 100 for i in range(n_dims)
    ]
    extreme_percentiles = [
        p
        for p in percentiles
        if p < (1 - percentile_allowance) or p > percentile_allowance
    ]

    if len(extreme_percentiles) == 0:
        print(
            "✅ PASS: All observed statistics fall well within the predictive distribution."
        )
    elif len(extreme_percentiles) <= max(
        1, n_dims // 5
    ):  # Allow some extreme values for larger dimensions
        print(
            "⚠️ WARNING: Some observed statistics are in the tails of the distribution."
        )
        print(
            f"  {len(extreme_percentiles)}/{n_dims} statistics have extreme percentiles: {[f'{p:.1f}%' for p in extreme_percentiles]}"
        )
        print("   Consider if the model/prior might need adjustment.")
    else:
        print("❌ FAIL: Many observed statistics fall outside the typical range.")
        print(
            f"   {len(extreme_percentiles)}/{n_dims} statistics have extreme percentiles."
        )
        print("   The model/prior may need to be reconsidered!")

    # Print detailed percentiles for reference
    print(f"\nDetailed percentiles for all {n_dims} statistics:")
    for _, (name, p) in enumerate(zip(stat_names, percentiles, strict=False)):
        status = (
            "🔴"
            if p < (1 - percentile_allowance) or p > (percentile_allowance)
            else "🟢"
        )
        if status == "🔴":
            print(f"{status} {name}: {p:.1f}%")

    return fig, axes


def plot_training_diagnostics(
    trainer: NeuralInference, title: str = "Neural Network Training Diagnostics"
) -> None:
    """
    Perform and visualize neural network training diagnostics.

    This function checks whether the neural density estimator was trained properly by:
    - Plotting training and validation loss curves
    - Checking for overfitting using relative metrics
    - Assessing convergence using adaptive thresholds

    Args:
        npe: Trained neural posterior estimator object with summary attribute
        title: Title for the plot
    """
    print("🔍 Neural Network Training Diagnostics")
    print("=" * 50)
    print("Checking if the neural network converged properly...\n")

    # Plot training summary
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Get the summary from the neural network
    summary = trainer.summary

    # summarize this into a list iter
    tags = ["training_loss", "validation_loss", "best_validation_loss"]
    assert summary is not None, "No summary available for training diagnostics."
    for tag in tags:
        assert tag in summary, f"No {tag} available for diagnostics."

    # Plot training and validation losses
    epochs = range(1, len(summary["training_loss"]) + 1)

    # Training loss
    ax = axes[0]
    ax.plot(epochs, np.array(summary["training_loss"]), label="Training", linewidth=2)
    if "validation_loss" in summary:
        ax.plot(
            epochs,
            np.array(summary["validation_loss"]),
            label="Validation",
            linewidth=2,
            alpha=0.7,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (negative log prob)")
    ax.set_title("Training and Validation Loss")
    ax.grid(True, alpha=0.3)

    # Best epoch
    if "best_validation_loss" in summary:
        best_epoch = summary["epochs_trained"]
        ax.axvline(
            best_epoch, color="green", linestyle="--", alpha=0.5, label="Best epoch"
        )

    ax.legend()

    # Loss difference (to check for overfitting)
    ax = axes[1]
    if "validation_loss" in summary:
        train_loss = -np.array(summary["training_loss"])
        val_loss = -np.array(summary["validation_loss"])
        loss_diff = val_loss - train_loss

        ax.plot(epochs, loss_diff, linewidth=2, color="orange")
        ax.axhline(0, color="black", linestyle="-", alpha=0.3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation - Training Loss")
        ax.set_title("Overfitting Check")
        ax.grid(True, alpha=0.3)

        # Mark concerning regions
        if len(loss_diff) > 0 and max(loss_diff) > 0:
            ax.fill_between(
                epochs,
                0,
                max(loss_diff) * 1.1,
                where=np.array(loss_diff) > 0.5,
                color="red",
                alpha=0.2,
                label="Potential overfitting",
            )

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

    # Assessment
    print("\n📊 Training Assessment:")
    print("-" * 40)
    final_train_loss = summary["training_loss"][-1]

    print(f"Final training loss: {final_train_loss:.3f}")

    final_val_loss = summary["validation_loss"][-1]
    print(f"Final validation loss: {final_val_loss:.3f}")

    # Use relative comparison for overfitting check
    if abs(final_train_loss) > 1e-6:  # Avoid division by zero
        relative_overfit = (final_val_loss - final_train_loss) / abs(final_train_loss)
        print(f"Relative overfitting metric: {relative_overfit:.1%}")

        if relative_overfit < 0.05:  # 5% relative increase
            print("✅ PASS: No signs of overfitting.")
        elif relative_overfit < 0.15:  # 15% relative increase
            print("⚠️  WARNING: Mild overfitting detected.")
        else:
            print("❌ FAIL: Significant overfitting detected!")
    else:
        print(
            "⚠️  WARNING: Training loss too close to zero for reliable overfitting assessment."
        )

    # Check convergence using adaptive threshold and relative measures
    total_epochs = len(summary["training_loss"])
    # Use at least 20 epochs or 20% of total training, whichever is larger
    min_epochs_for_check = max(20, total_epochs // 5)

    if total_epochs >= min_epochs_for_check:
        recent_losses = summary["training_loss"][-min_epochs_for_check:]
        loss_mean = np.mean(recent_losses)
        loss_std = np.std(recent_losses)

        # Use coefficient of variation (relative standard deviation)
        if abs(loss_mean) > 1e-6:  # Avoid division by zero
            cv = loss_std / abs(loss_mean)
            print(
                f"Loss coefficient of variation (last {min_epochs_for_check} epochs): {cv:.1%}"
            )

            if cv < 0.01:  # 1% relative variation
                print("✅ Training converged (stable loss).")
            elif cv < 0.05:  # 5% relative variation
                print("⚠️  Training mostly converged but still shows some variation.")
            else:
                print("⚠️  Training may not have fully converged.")
        else:
            print(
                "⚠️  WARNING: Training loss too close to zero for reliable convergence assessment."
            )
    else:
        print(
            f"⚠️  Too few epochs ({total_epochs}) for reliable convergence assessment. Need at least {min_epochs_for_check}."
        )
