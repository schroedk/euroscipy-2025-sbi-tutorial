# EuroSciPy 2025 SBI Tutorial - Slide Outline for Keynote

## Slide Deck Structure

**Total slides: ~30-35**
**Font requirements: Min 17pt regular, 14pt for footnotes**
**Color scheme: Use color-blind friendly palette (e.g., Okabe-Ito)**

---

## PART 1: WHY SBI? (15 minutes, ~8 slides)

### Slide 1: Title Slide

**Title:** Quantifying Uncertainty in Scientific Simulators with Neural Networks
**Subtitle:** A Hands-On Introduction to Simulation-Based Inference
**Content:**

- EuroSciPy 2025, Kraków
- Instructor names and affiliations
- QR code/link to materials: github.com/[repo]
- Large sbi logo

### Slide 2: The Environmental Monitoring Challenge

**Title:** Your Mission: Wildlife Population Monitoring
**Visual:** Dashboard showing monthly summary statistics
**Content:**
- "You're a data scientist for a regional environmental agency"
- "Task: Monitor wolf 🐺 and deer 🦌 populations with limited resources"
- "Data: Monthly reports with average & peak populations only"
- "Challenge: Infer continuous dynamics from sparse summaries"
**Animation:** Show monthly reports accumulating over time

### Slide 3: The Traditional Approach

**Title:** Finding Parameters: The Optimization Way
**Visual:** Split screen showing:

- Left: Parameter space with single point (best fit)
- Right: Simulated vs observed data
**Content:**
- "Grid search or optimization finds ONE best parameter set"
- "θ* = argmin ||simulator(θ) - observation||"
- ✅ Gives an answer
- ❌ No uncertainty quantification
- ❌ Misses equally good parameters
**Animation:** Show optimization converging to single point

### Slide 4: The Hidden Problem

**Title:** Many Parameters Can Explain Your Data!
**Visual:** Same observation, but now showing multiple parameter combinations
**Content:**
- Show 3 different parameter sets
- All produce similar observations
- "Which one is correct?"
- "What about predictions?"
**Animation:** Morph between different simulations from different parameters

### Slide 5: What We Really Want

**Title:** From Point Estimates to Distributions
**Visual:** Transform from single point to probability distribution
**Content:**
- Left: Single point estimate ❌
- Right: Posterior distribution ✅
- "p(parameters | observation)"
- Shows uncertainty
- Reveals parameter correlations
- Enables robust predictions
**Animation:** Point expanding into distribution

### Slide 6: The Likelihood Problem

**Title:** Why Can't We Just Use Bayes' Rule?
**Visual:** Bayes' rule with crossed-out likelihood
**Content:**
- p(θ|x) ∝ p(x|θ) × p(θ)
- "For complex simulators:"
  - Stochastic: Different output each run
  - Black-box: No analytical likelihood
  - Slow: Can't evaluate millions of times
- "Examples: Climate models, epidemics, neural circuits"

### Slide 7: Enter Simulation-Based Inference

**Title:** SBI: Let Neural Networks Learn from Simulations
**Visual:** Flow diagram: Parameters → Simulator → Data → Neural Network → Posterior
**Content:**
- "Core idea: Learn from simulated examples"
- "No likelihood needed!"
- "Works with any simulator"
- Modern ML meets Bayesian inference
**Animation:** Flow through the pipeline

### Slide 8: What You'll Learn Today

**Title:** From Theory to Your Own Problems in 90 Minutes
**Visual:** Three connected boxes showing progression
**Content:**
1. **Quick Win (15 min)**: Run your first inference
2. **Trust (20 min)**: Verify your results
3. **Apply (20 min)**: Your own simulator
- "All code provided, focus on understanding"
- "Bring your questions!"

---

## PART 2: CORE INTUITION (10 minutes, ~6 slides)

### Slide 9: Two Approaches to SBI

**Title:** Classical vs Modern SBI
**Visual:** Side-by-side comparison
**Content:**
- **Classical: Rejection Sampling**
  - Simple, intuitive
  - Inefficient for high dimensions
- **Modern: Neural Density Estimation**
  - Efficient, scalable
  - Powers the sbi package
**Note:** "We'll see both for intuition"

### Slide 10: Rejection Sampling in 5 Lines

**Title:** The Simplest SBI Algorithm
**Visual:** Code on left, visualization on right
**Code (large font):**
```python
# 1. Sample parameters from prior
θ ~ prior()
# 2. Simulate data
x_sim = simulator(θ)
# 3. Accept if close to observation
if distance(x_sim, x_obs) < ε:
    accept θ
```
**Visualization:** Animation showing accept/reject

### Slide 11: Rejection Sampling Problem
**Title:** The Curse of Dimensionality
**Visual:** Acceptance rate dropping with dimensions
**Content:**
- 2D: 10% acceptance ✅
- 5D: 0.1% acceptance 😐
- 10D: 0.00001% acceptance ❌
- "Need millions of simulations!"
**Animation:** Shrinking acceptance region

### Slide 12: Neural Posterior Estimation (NPE)
**Title:** Learning to Predict Parameters from Data
**Visual:** Neural network architecture (simplified)
**Content:**
- Input: Observed data x
- Output: Distribution over parameters θ
- "Train once, reuse for any observation!"
- Amortized inference
**Key insight:** "Turn inference into supervised learning"

### Slide 13: How NPE Works
**Title:** Training a Neural Density Estimator
**Visual:** Training loop animation
**Steps:**
1. Generate training data: {(θᵢ, xᵢ)}
2. Train network: q(θ|x) ≈ p(θ|x)
3. At test time: Input x_obs → Get posterior
**Content:**
- "Like training an image classifier, but output is a distribution"
- Normalizing flows for flexible distributions

### Slide 14: The Power of Amortization
**Title:** Train Once, Infer Many Times
**Visual:** Comparison table
**Content:**
| Method | New observation | Cost |
|--------|----------------|------|
| MCMC | Re-run everything | Hours |
| Rejection | Re-run everything | Hours |
| NPE | Forward pass | Milliseconds |
**Bottom text:** "Perfect for real-time applications!"

---

## PART 3: HANDS-ON TRANSITION (2 minutes, 2 slides)

### Slide 15: Let's Code!
**Title:** Three Exercises, Progressive Difficulty
**Visual:** Three notebook icons with checkmarks
**Content:**
1. 📓 **First Inference**: Lotka-Volterra with 5 lines
2. 🔍 **Diagnostics**: Is my posterior trustworthy?
3. 🚀 **Your Problem**: Template for any simulator
**Note:** "Solutions available if you get stuck"

### Slide 16: Quick Setup Check
**Title:** Everyone Ready?
**Visual:** Terminal/notebook screenshot
**Content:**
```python
import sbi
import torch
print("Ready for SBI! 🚀")
```
- "Raise hand if you see errors"
- "Helpers available"
- "Colab backup: [link]"

---

## PART 4: EXERCISE SUPPORT SLIDES (Reference during coding)

### Slide 17: Exercise 1 Overview
**Title:** Your First Inference - Lotka-Volterra
**Visual:** Expected output plots
**Key steps:**
1. Load simulator
2. Define prior
3. Run inference
4. Visualize posterior
**Time:** 15 minutes

### Slide 18: Exercise 2 Overview
**Title:** Diagnostic Tools
**Visual:** Example diagnostic plots
**Tools:**
- Posterior predictive check
- Coverage test
- Warning signs to watch
**Time:** 20 minutes

### Slide 19: Exercise 3 Overview
**Title:** Your Own Simulator
**Visual:** Template structure
**Options:**
- Your simulator (if brought)
- Ball throw example
- SIR epidemic model
**Time:** 20 minutes

---

## PART 5: NEXT STEPS (5 minutes, ~5 slides)

### Slide 20: Beyond NPE

**Title:** The SBI Toolbox
**Visual:** Method comparison chart
**Content:**
| Method | Learns | Best for |
|--------|--------|----------|
| NPE | p(θ\|x) | Fast amortized inference |
| NLE | p(x\|θ) | MCMC sampling |
| NRE | p(θ,x)/p(θ)p(x) | Model comparison |
| Sequential | Iteratively | Sample efficiency |

### Slide 21: Common Pitfalls
**Title:** Learn from Our Mistakes
**Visual:** Warning signs icons
**Content:**
1. ⚠️ **Prior too wide**: Wasted simulations
2. ⚠️ **Too few simulations**: Poor approximation
3. ⚠️ **Ignoring diagnostics**: False confidence
4. ⚠️ **Complex summary stats**: Information loss
**Bottom:** "Always validate your results!"

### Slide 22: Advanced Topics
**Title:** Where to Go Deeper
**Visual:** Topic tree/mindmap
**Branches:**
- Multi-round inference (sequential)
- Model comparison
- Embedding networks
- Likelihood-based methods
- Compositional inference
**Resources:** Links to papers and tutorials

### Slide 23: The SBI Community

**Title:** Join Us!
**Visual:** Community logos and screenshots
**Content:**

- 📦 Package: github.com/sbi-dev/sbi
- 💬 Discussions: GitHub Discussions
- 📚 Docs: sbi-dev.github.io/sbi
- 🎓 Papers: JOSS, NeurIPS, ICML
- 🤝 Contributors: 70+ and growing!
**QR code:** Link to GitHub

### Slide 24: Real-World Applications

**Title:** SBI in the Wild
**Visual:** Grid of application domains
**Examples:**

- 🧠 Neuroscience: Neural circuit models
- 🦠 Epidemiology: COVID-19 modeling
- 🌍 Climate: Weather prediction
- 🔬 Physics: Gravitational waves
- 🧬 Biology: Gene regulation
**Bottom:** "Your application next?"

### Slide 25: Thank You & Q&A

**Title:** Questions?
**Visual:** Contact information and resources
**Content:**

- Tutorial materials: [GitHub link]
- Email: [instructor emails]
- Office hours: [if applicable]
- Feedback form: [QR code]
**Large text:** "What will you infer?"

---

## BACKUP SLIDES (If time/questions)

### Slide B1: Mathematical Details

**Title:** For the Curious: NPE Loss Function
**Content:** Mathematical formulation of normalizing flows

### Slide B2: Benchmark Results

**Title:** SBI Performance Comparisons
**Visual:** Benchmark charts from papers

### Slide B3: Installation Troubleshooting

**Title:** Common Setup Issues
**Content:** Solutions for typical problems

### Slide B4: GPU Acceleration

**Title:** Scaling to Large Problems
**Content:** Tips for GPU usage

---

## Presentation Notes

### Visual Design Guidelines

- **Color palette**: Use Okabe-Ito or Viridis (color-blind safe)
- **Font sizes**:
  - Titles: 36-44pt
  - Body: 20-24pt
  - Code: 18-20pt (monospace)
  - Footnotes: 14-16pt minimum
- **Animations**: Keep simple, avoid flashing
- **Code highlighting**: Use syntax colors with good contrast

### Speaker Notes Suggestions

- Slide 2: Mention actual forest management challenges in Poland
- Slide 6: Have backup explanation for non-Bayesian audience
- Slide 10: Live code this if possible
- Slide 15: Check everyone has notebooks open
- Slide 23: Mention upcoming sbi hackathon/workshop

### Accessibility Reminders

- Describe visualizations verbally
- Pause before transitions
- Repeat questions from audience
- Provide alt-text for all images
- Ensure QR codes also have text URLs
