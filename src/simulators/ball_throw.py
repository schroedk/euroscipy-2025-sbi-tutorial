"""
Ball Throw Simulator for SBI Tutorial
=====================================

This simulator models projectile motion with air resistance, suitable for
demonstrating SBI on a physics problem.

The ball's trajectory follows these differential equations:
- Horizontal: d²x/dt² = wind - friction·dx/dt
- Vertical: d²y/dt² = -gravity - friction·dy/dt

This represents a ball thrown at an angle, affected by gravity, air resistance,
and optionally wind.
"""

import numpy as np
import torch
from sbi.utils import BoxUniform


def ball_throw_simulator(
    params: torch.Tensor | np.ndarray, return_trajectory: bool = False
) -> torch.Tensor | tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    Simulate a ball throw with air resistance.

    This models the trajectory of a ball thrown at an angle, affected by
    gravity, wind, and air friction. The physics follows:

    d²x/dt² = W - μ·dx/dt  (horizontal: wind - friction)
    d²y/dt² = -g - μ·dy/dt (vertical: gravity - friction)

    Parameters
    ----------
    params : torch.Tensor or np.ndarray with shape (3,) or (4,)
        If 3 parameters:
            [0] initial_velocity: Initial speed in m/s, range [5, 30]
            [1] launch_angle: Launch angle in radians, range [0.2, 1.4]
            [2] friction_coef: Air friction coefficient, range [0.0, 0.5]
        If 4 parameters (includes wind):
            [3] wind_speed: Horizontal wind in m/s, range [-5, 5]

    return_trajectory : bool, optional
        If True, also returns full trajectory for visualization

    Returns
    -------
    observations : torch.Tensor with shape (2,)
        [0] landing_distance: Horizontal distance where ball hits ground (meters)
        [1] max_height: Maximum height reached during flight (meters)

    If return_trajectory=True, also returns:
        x_trajectory : np.ndarray - x positions over time
        y_trajectory : np.ndarray - y positions over time

    Examples
    --------
    >>> params = torch.tensor([15.0, 0.8, 0.1])  # 15 m/s, 45°, low friction
    >>> observations = ball_throw_simulator(params)
    >>> print(f"Landing: {observations[0]:.1f}m, Max height: {observations[1]:.1f}m")
    """
    # Convert to numpy for simulation
    if isinstance(params, torch.Tensor):
        params_np = params.detach().cpu().numpy()
    else:
        params_np = np.array(params)

    # Extract parameters
    v0 = params_np[0]  # Initial velocity
    angle = params_np[1]  # Launch angle (radians)
    friction = params_np[2]  # Friction coefficient
    wind = params_np[3] if len(params_np) > 3 else 0.0  # Wind (optional)

    # Physical constants
    g = 9.81  # Gravitational acceleration (m/s²)
    dt = 0.01  # Time step for integration (seconds)

    # Initial conditions
    x, y = 0.0, 0.0
    vx = v0 * np.cos(angle)
    vy = v0 * np.sin(angle)

    # Storage for trajectory (if needed)
    if return_trajectory:
        x_traj, y_traj = [x], [y]

    # Track observables
    max_height = 0.0

    # Simulate until ball hits ground (with safety limit)
    for _ in range(10000):
        # Update velocities (Euler method with friction and forces)
        vx_new = vx + (wind - friction * vx) * dt
        vy_new = vy + (-g - friction * vy) * dt

        # Update positions
        x_new = x + vx * dt
        y_new = y + vy * dt

        # Check if ball hit ground
        if y_new < 0:
            # Interpolate to find exact landing point
            t_impact = -y / vy if vy != 0 else 0
            landing_distance = x + vx * t_impact
            break

        # Update state
        x, y = x_new, y_new
        vx, vy = vx_new, vy_new
        max_height = max(max_height, y)

        if return_trajectory:
            x_traj.append(x)
            y_traj.append(y)
    else:
        # Safety fallback if simulation doesn't converge
        landing_distance = x

    # Add realistic observation noise (5% relative error)
    noise_scale = 0.05
    landing_distance *= 1 + np.random.randn() * noise_scale
    max_height *= 1 + np.random.randn() * noise_scale

    # Ensure positive values
    landing_distance = max(0.1, landing_distance)
    max_height = max(0.1, max_height)

    # Prepare output
    observations = torch.tensor([landing_distance, max_height], dtype=torch.float32)

    if return_trajectory:
        return observations, np.array(x_traj), np.array(y_traj)
    return observations


def create_ball_throw_prior(include_wind: bool = False):
    """
    Create prior distribution for ball throw parameters.

    Parameters
    ----------
    include_wind : bool
        If True, includes wind as 4th parameter

    Returns
    -------
    prior : sbi.utils.BoxUniform
        Prior distribution over parameters
    """

    if include_wind:
        # With wind: [velocity, angle, friction, wind]
        low = torch.tensor([5.0, 0.2, 0.0, -5.0])
        high = torch.tensor([30.0, 1.4, 0.5, 5.0])
    else:
        # Without wind: [velocity, angle, friction]
        low = torch.tensor([5.0, 0.2, 0.0])
        high = torch.tensor([30.0, 1.4, 0.5])

    return BoxUniform(low=low, high=high)


def generate_observation(true_params: None | torch.Tensor = None, seed: int = 42):
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
        # Default parameters: moderate throw
        true_params = torch.tensor([15.0, 0.8, 0.1])  # 15 m/s, ~45°, low friction

    observations = ball_throw_simulator(true_params)
    return observations, true_params


# Quick test
if __name__ == "__main__":
    print("Testing Ball Throw Simulator...")

    # Test basic throw
    params = torch.tensor([20.0, 0.7, 0.2])  # 20 m/s, ~40°, medium friction
    obs = ball_throw_simulator(params)
    print(f"Input: v={params[0]:.1f} m/s, θ={params[1]:.2f} rad, μ={params[2]:.2f}")
    print(f"Output: distance={obs[0]:.1f}m, max_height={obs[1]:.1f}m")

    # Test with trajectory
    obs, x_traj, y_traj = ball_throw_simulator(params, return_trajectory=True)
    print(f"Trajectory: {len(x_traj)} time steps")

    # Test prior
    prior = create_ball_throw_prior()
    print(f"Prior created: {prior}")

    print("\n✅ Ball throw simulator ready for tutorial!")
