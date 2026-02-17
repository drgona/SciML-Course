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
