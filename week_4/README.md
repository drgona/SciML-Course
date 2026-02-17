# Week 4: Neural Ordinary Differential Equations (NODEs)

> <p align="center">
>   <samp>
>     The future is not a straight line. There are many different pathways.
>   </samp>
> </p>
> <p align="right">
>   <samp>— <strong>Katsuhiro Otomo</strong>, <em>Akira</em></samp>
> </p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/marimo-team/marimo/raw/main/docs/_static/marimo-logotype-thick.svg">
    <img src="https://github.com/marimo-team/marimo/raw/main/docs/_static/marimo-logotype-horizontal.png" alt="Marimo" height="50" />
  </picture>
  &nbsp;&nbsp;&nbsp;
  <img src="https://github.com/google/jax/raw/main/images/jax_logo_250px.png" alt="JAX" height="50" />
  &nbsp;&nbsp;&nbsp;
  <img src="https://raw.githubusercontent.com/pytorch/pytorch/main/docs/source/_static/img/pytorch-logo-dark.png" alt="PyTorch" height="50" />
</p>

Week 4 course materials for training Neural ODE models for pendulum dynamics using both backpropagation through time (BPTT) and adjoint-style gradients.

## General Problem Formulation

Given trajectory data from an unknown continuous-time system, we want to learn a neural vector field that reproduces the observed dynamics.

For each trajectory \(m\), with initial condition \(x_0^{(m)}\), the true system evolves as
\[
\dot{x}(t) = g(x(t), t), \qquad x(t_0) = x_0^{(m)},
\]
and observations are available on a time grid \(\{t_k\}_{k=0}^{N-1}\):
\[
y_k^{(m)} \approx x(t_k; x_0^{(m)}).
\]

We model dynamics with a Neural ODE:
\[
\dot{x}_\theta(t) = f_\theta(x_\theta(t), t), \qquad x_\theta(t_0) = x_0^{(m)},
\]
and obtain predictions by numerically integrating \(f_\theta\) (RK4 in this week’s scripts/notebooks).

Training solves
\[
\min_\theta \frac{1}{MN}\sum_{m=1}^{M}\sum_{k=0}^{N-1}
\left\|x_\theta(t_k; x_0^{(m)}) - y_k^{(m)}\right\|_2^2 + \lambda\,\mathcal{R}(\theta),
\]
with gradients computed either by:
- BPTT: differentiate through all solver steps.
- Adjoint: use a backward-time sensitivity solve.

## Notations

- \(x(t) \in \mathbb{R}^d\): system state at time \(t\).
- \(x_0^{(m)}\): initial state for trajectory \(m\).
- \(g(\cdot)\): unknown true dynamics generating data.
- \(f_\theta(\cdot)\): neural vector field (NODE) with parameters \(\theta\).
- \(\theta\): trainable model parameters.
- \(t_0, \dots, t_{N-1}\): discrete time grid used for simulation/training.
- \(M\): number of trajectories in a batch/dataset.
- \(N\): number of time samples per trajectory.
- \(y_k^{(m)}\): observed state for trajectory \(m\) at time \(t_k\) (possibly noisy).
- \(x_\theta(t_k; x_0^{(m)})\): model-predicted state at time \(t_k\).
- \(\mathcal{L}(\theta)\): training objective (data mismatch + optional regularization).
- \(\mathcal{R}(\theta)\): regularization term (for example, weight decay or smoothness-related penalties).
- \(\lambda\): regularization weight.

## Interactive Notebooks

### <img src="https://github.com/marimo-team/marimo/raw/main/docs/_static/marimo-logotype-horizontal.png" alt="marimo" height="40" /> + <img src="https://github.com/google/jax/raw/main/images/jax_logo_250px.png" alt="JAX" height="30" />

- [![Open in molab](https://molab.marimo.io/molab-shield.svg)](https://molab.marimo.io/notebooks/nb_Qb9QbK24mhGmibu21nnK1B) Neural ODE BPTT (JAX)
- [![Open in molab](https://molab.marimo.io/molab-shield.svg)](https://molab.marimo.io/notebooks/nb_WwLinsU3xnFY6kP3MzbEva) Neural ODE Adjoint (JAX)

### <img src="https://github.com/marimo-team/marimo/raw/main/docs/_static/marimo-logotype-horizontal.png" alt="marimo" height="40" /> + <img src="https://raw.githubusercontent.com/pytorch/pytorch/main/docs/source/_static/img/pytorch-logo-dark.png" alt="PyTorch" height="30" />

- [![Open in molab](https://molab.marimo.io/molab-shield.svg)](https://molab.marimo.io/notebooks/nb_zX5adqKmNem4MjViWf5Uxc) Neural ODE BPTT (PyTorch)
- [![Open in molab](https://molab.marimo.io/molab-shield.svg)](https://molab.marimo.io/notebooks/nb_jd47abUL1WHg6FH55T5QEy) Neural ODE Adjoint (PyTorch)

## Local Run (Optional)

If hosted notebooks feel slow, run locally with marimo or use script versions in `week_4/scripts`.

```bash
# Option 1: marimo app
marimo run week_4/marimo/node_jax/node_bptt_jax.py

# Option 2: marimo editor mode
marimo edit week_4/marimo/node_jax/node_bptt_jax.py

# Option 3: script fallback
python week_4/scripts/jax/NODE_BPTT_vanilla.py
```

## Week 4 Structure Map

```text
week_4/
├── README.md                           # Week 4 landing page (this file)
├── Week_4_nodes.pptx                   # Lecture slides
├── node training strategies.pdf        # Reading/reference notes
├── marimo/                             # Interactive notebook-style apps
│   ├── node_jax/                       # JAX + marimo apps
│   │   ├── node_bptt_jax.py            # BPTT NODE (vanilla defaults + tuning)
│   │   └── node_adjoint_jax.py         # Adjoint NODE
│   └── node_torch/                     # PyTorch + marimo apps
│       ├── node_bptt_torch.py          # BPTT NODE (vanilla defaults + tuning)
│       └── node_adjoint_torch.py       # Adjoint NODE
└── scripts/                            # Standalone Python scripts
    ├── jax/                            # JAX script implementations
    │   ├── NODE_BPTT_vanilla.py
    │   ├── NODE_BPTT_tuned.py
    │   └── NODE_adjoint_example.py
    └── torch/                          # PyTorch script implementations
        ├── NODE_BPTT_vanilla.py
        ├── NODE_BPTT_tuned.py
        └── NODE_adjoint_example.py
```
