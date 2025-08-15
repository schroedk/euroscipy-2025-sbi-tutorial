# Simulators for SBI Tutorial

This directory contains three example simulators for the EuroSciPy 2025 SBI tutorial. Each demonstrates a different scientific domain and complexity level.

## 🐺🦌 Lotka-Volterra Model (`lotka_volterra.py`)

**Domain**: Ecology / Population dynamics
**Story**: Environmental monitoring of wolf and deer populations
**Complexity**: Medium (2 coupled ODEs)

- **Parameters**: Birth rates, predation rates (4 params)
- **Observations**: Population summary statistics over time
- **Use case**: Main tutorial example (Exercises 1 & 2)

## 🎾 Ball Throw Physics (`ball_throw.py`)

**Domain**: Physics / Projectile motion
**Story**: Analyzing sports trajectories (baseball, golf)
**Complexity**: Simple (basic physics with drag)

- **Parameters**: Initial velocity, angle, friction (3-4 params)
- **Observations**: Landing distance, maximum height
- **Use case**: Quick, intuitive example for Exercise 3

## 🦠 SIR Epidemic Model (`sir_model.py`)

**Domain**: Epidemiology / Disease spread
**Story**: Tracking disease outbreak in a community
**Complexity**: Medium (3 coupled ODEs)

- **Parameters**: Infection rate, recovery rate, initial infected (3 params)
- **Observations**: Peak infected, time to peak, total recovered, duration
- **Use case**: Alternative example for Exercise 3

## Usage

### Quick Start

```python
import torch
from simulators.ball_throw import ball_throw_simulator, create_ball_throw_prior

# Define parameters
params = torch.tensor([15.0, 0.8, 0.1])  # velocity, angle, friction

# Run simulation
observations = ball_throw_simulator(params)
print(f"Ball lands at {observations[0]:.1f}m, max height {observations[1]:.1f}m")

# Create prior for SBI
prior = create_ball_throw_prior()
```

### For SBI Workflow

```python
from sbi import inference
from simulators.sir_model import sir_epidemic_simulator, create_sir_prior

# Setup
prior = create_sir_prior()
simulator = sir_epidemic_simulator

# Run NPE
npe = inference.NPE(prior)
npe.append_simulations(simulator, num_simulations=5000).train()
posterior = npe.build_posterior()

# Inference
observed_data = torch.tensor([2500, 45, 8000, 120])  # Example observation
samples = posterior.sample((1000,), x=observed_data)
```

### Visualization Support

Each simulator can return additional data for visualization:

```python
# Ball trajectory
obs, x_traj, y_traj = ball_throw_simulator(params, return_trajectory=True)

# Epidemic time series
obs, time_series = sir_epidemic_simulator(params, return_time_series=True)

# Plot results
import matplotlib.pyplot as plt
plt.plot(time_series['t'], time_series['I'], label='Infected')
plt.show()
```

## Design Principles

All simulators follow these conventions:

1. **Input**: Parameters as `torch.Tensor` or `numpy.ndarray`
2. **Output**: Observations as `torch.Tensor`
3. **Noise**: Realistic observation noise included (5-10%)
4. **Speed**: Fast execution (< 0.1s per simulation)
5. **Priors**: Helper functions to create appropriate priors
6. **Documentation**: Comprehensive docstrings with examples

## Choosing a Simulator

- **For learning SBI basics**: Start with Lotka-Volterra (main tutorial)
- **For quick experiments**: Use ball throw (simple, intuitive)
- **For epidemiology interest**: Try SIR model (timely, relevant)
- **For your research**: Adapt these as templates for your own simulators

## Requirements

- `torch >= 2.0.0`
- `numpy >= 1.24.0`
- `scipy >= 1.10.0` (for Lotka-Volterra ODE integration)
- `sbi >= 0.23.0` (for prior creation)
