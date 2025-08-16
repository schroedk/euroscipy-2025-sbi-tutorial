# EuroSciPy 2025: Simulation-Based Inference Tutorial

**Beyond Likelihoods: Bayesian Parameter Inference for Black-Box Simulators with sbi**

📍 Kraków, Poland | 90 minutes | Intermediate Track

## 🎯 Learning Objectives

By the end of this tutorial, you will:

- Understand when and why to use Simulation-Based Inference (SBI)
- Run parameter inference for any Python simulator using the `sbi` package
- Diagnose whether your inference results are trustworthy
- Apply SBI to your own scientific problems

## 📚 Pre-Tutorial Homework (Required)

### 1. Environment Setup

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone this repository
git clone https://github.com/[YOUR-REPO]/euroscipy-2025-sbi-tutorial.git
cd euroscipy-2025-sbi-tutorial

# Create environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

### 2. Test Your Setup
```python
# Run this to verify installation
python -c "import sbi; import torch; print('✅ Setup complete!')"
```

### 3. Optional but Recommended

- **Read a bit about SBI**:
  - [Last year's EuroSciPy Talk](https://pretalx.com/euroscipy-2024/talk/893KBK/), [slides](materials/2024-08-30-EuroSciPy2024-SBI.pdf)
  - [Sbi overview blog post](https://transferlab.ai/series/simulation-based-inference/)
  - [Sbi software paper](https://joss.theoj.org/papers/10.21105/joss.07754)
  
- **Bring Your Simulator**: If you have a scientific simulator you'd like to apply SBI to, prepare a simplified version that:
  - Takes parameters as input (numpy array or torch tensor)
  - Returns observations as output (numpy array or torch tensor)
  - Runs in < 1 second per simulation
  - Example: `def simulator(params): return observations`

### 4. Backup Option

If you have any issues with local setup, you can use our Google Colab notebooks (links will be provided at the tutorial).

## 📋 Tutorial Outline

### Part 1: Why SBI? (15 min)

- The Environmental Monitoring Challenge
- Point estimates vs. uncertainty quantification
- When traditional methods fail

### Part 2: Core Intuition (10 min)

- Rejection sampling in 5 lines of code
- Neural Posterior Estimation concept
- Learning parameter-data relationships

### Part 3: Hands-On Exercises (55 min)

#### Exercise 1: Your First Inference (15 min)

- Lotka-Volterra predator-prey model
- Running NPE with `sbi`
- Visualizing posterior distributions

#### Exercise 2: Trust but Verify (20 min)

- Posterior predictive checks
- Coverage diagnostics
- Interpreting warning signs

#### Exercise 3: Your Own Problem (20 min)

- Adapt template to your simulator
- OR use provided examples (ball throw, SIR model)
- Specify priors and run inference

### Part 4: Next Steps (5 min)

- Advanced SBI methods (NLE, NRE, sequential)
- Resources and community
- Common pitfalls to avoid

### Q&A (5 min)

## 🗂️ Repository Structure

```
euroscipy-2025-sbi-tutorial/
├── README.md                    # This file
├── pyproject.toml               # Project dependencies
├── slides/                      # Presentation slides
│   └── sbi_tutorial.md
├── src/
│   ├── 00_setup_test.py       # Verify installation
│   ├── 01_first_inference.ipynb    # Exercise 1
│   ├── 02_diagnostics.ipynb        # Exercise 2
│   ├── 03_your_problem.ipynb       # Exercise 3
│   └── simulators/             # Example simulators
│       ├── lotka_volterra.py
│       ├── ball_throw.py
│       └── sir_model.py
└── materials/
    ├── 2024-08-30-EuroSciPy2024-SBI.pdf  # Optional pre-reading
    └── references.md                     # Further resources, TBD.

```

## 💻 Technical Requirements

- Python 3.10+
- Basic familiarity with PyTorch tensors (we'll provide refresher)
- Understanding of your own simulator (if bringing one)

## ♿ Accessibility

- All materials use color-blind friendly palettes
- Code uses font size ≥14pt, recommended 17pt
- Slides available for download at tutorial start
- Solutions provided for all exercises

## 👥 Instructors

TBD.

## 📧 Support

- **Before the tutorial**: Open an issue on GitHub
- **During the tutorial**: Raise your hand or ask in chat
- **After the tutorial**: Join the sbi community on [GitHub Discussions](https://github.com/sbi-dev/sbi/discussions)

## 🔗 Links

- [sbi documentation](https://sbi.readthedocs.io/en/latest/)
- [sbi paper](https://joss.theoj.org/papers/10.21105/joss.07754)
- [sbi overview blog post](https://transferlab.ai/series/simulation-based-inference/)
- [Tutorial paper (preprint)](https://arxiv.org/abs/[LINK])

## 📝 License

This tutorial is licensed under CC BY 4.0. Feel free to reuse and adapt with attribution.
