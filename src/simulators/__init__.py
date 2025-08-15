"""
Simulators for EuroSciPy 2025 SBI Tutorial
==========================================

This package contains example simulators for demonstrating Simulation-Based Inference:

1. **Lotka-Volterra**: Predator-prey population dynamics (wolves & deer)
2. **Ball Throw**: Projectile motion with air resistance
3. **SIR Model**: Epidemic spread through a population

Each simulator:
- Takes parameters as input (torch.Tensor or numpy array)
- Returns observations/summary statistics (torch.Tensor)
- Includes realistic observation noise
- Runs quickly for interactive tutorials

Examples
--------
>>> from simulators.ball_throw import ball_throw_simulator, create_ball_throw_prior
>>> params = torch.tensor([15.0, 0.8, 0.1])
>>> observations = ball_throw_simulator(params)

>>> from simulators.sir_model import sir_epidemic_simulator, create_sir_prior
>>> params = torch.tensor([0.5, 0.1, 10])
>>> observations = sir_epidemic_simulator(params)

>>> from simulators.lotka_volterra import lotka_volterra_simulator, create_lotka_volterra_prior
>>> params = torch.tensor([0.5, 0.025, 0.01, 0.5])
>>> observations = lotka_volterra_simulator(params)
"""

# Make simulators easily importable
from .ball_throw import ball_throw_simulator, create_ball_throw_prior
from .ball_throw import generate_observation as generate_ball_data
from .lotka_volterra import (
    create_lotka_volterra_prior,
    lotka_volterra_simulator,
    simulate,
)
from .sir_model import calculate_R0, create_sir_prior, sir_epidemic_simulator
from .sir_model import generate_observation as generate_sir_data

__all__ = [
    # Lotka-Volterra
    "lotka_volterra_simulator",
    "create_lotka_volterra_prior",
    "simulate",
    # Ball throw
    "ball_throw_simulator",
    "create_ball_throw_prior",
    "generate_ball_data",
    # SIR model
    "sir_epidemic_simulator",
    "create_sir_prior",
    "calculate_R0",
    "generate_sir_data",
]
