# /// script
# [tool.marimo.runtime]
# auto_instantiate = false
# on_cell_change = "lazy"
# ///

import marimo

__generated_with = "0.19.9"
app = marimo.App(
    app_title="Neural ODE Adjoint (Torch)",
    auto_download=["html"],
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Neural ODE Training with a Custom Adjoint Solver (PyTorch)

    This notebook converts `week_4/scripts/torch/NODE_adjoint_example.py` into an interactive didactical notebook.

    **Navigation:** [Jump to Control Panel](#control-panel)
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn as nn

    return mo, nn, np, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 1. Problem Formulation

    We observe noisy pendulum trajectories and train a Neural ODE model.

    Reference dynamics:

    $$x(t)=\begin{bmatrix}u(t)\\v(t)\end{bmatrix},\quad
    \dot{u}=v,\quad
    \dot{v}=-\beta v-\frac{g}{\ell}\sin(u).$$

    Learned dynamics:

    $$\frac{dx}{dt}=f_\theta(x),\qquad x(t_0)=x_0.$$

    Training objective:

    $$\mathcal{L}(\theta)=\frac{1}{MN}\sum_{i=1}^{M}\sum_{k=0}^{N}\|x^i_k-\hat{x}^i_k\|_2^2 + \lambda R(\theta).$$

    Write the sampled loss as

    $$\mathcal{L}(\theta)=\sum_{k=0}^{N}\ell_k(x_k,\theta)+\lambda R(\theta),\qquad
    \ell_k=\frac{1}{M}\sum_{i=1}^{M}\|x_k^i-\hat{x}_k^i\|_2^2.$$

    Instead of storing every solver operation like BPTT, the adjoint method tracks
    two continuous sensitivity states:

    $$\frac{da}{dt}=-(\nabla_x f_\theta)^\top a,\qquad
    \frac{dg}{dt}=-(\nabla_\theta f_\theta)^\top a,\qquad
    \nabla_\theta\mathcal{L}=g(t_0).$$

    Boundary conditions for backward integration are

    $$a(T)=\nabla_{x_N}\ell_N,\qquad g(T)=0.$$

    For losses sampled at grid points, the state sensitivity receives jump updates:

    $$a(t_k^-)=a(t_k^+)+\nabla_{x_k}\ell_k.$$

    Over one interval, parameter-gradient accumulation is

    $$g(t_{k-1})=g(t_k)+\int_{t_k}^{t_{k-1}}
    -(\nabla_\theta f_\theta(x(t)))^\top a(t)\,dt.$$

    Notation used in this notebook:

    - $t$: continuous time, with $t_0$ (start), $T$ (end), and samples $t_k$.
    - $x(t)\in\mathbb{R}^n$: state, $x_k\approx x(t_k)$ on the grid, $x_0=x(t_0)$.
    - $f_\theta$: neural vector field with parameters $\theta\in\mathbb{R}^p$.
    - $a(t)$: state sensitivity (adjoint), $g(t)$: parameter-gradient accumulator.
    - $M$: number of trajectories, $N$: number of time steps.
    - $\hat{x}_k^i$: observation for trajectory $i$ at index $k$.

    Note: in the adjoint equations, $g(t)$ is a gradient accumulator; it is not
    the pendulum gravity constant.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.1 Initial condition sampler $x_0$

    For each trajectory $i$:

    $$u_0^i \sim \mathcal{U}[-0.5,0.5],\qquad
    v_0^i \sim \mathcal{U}[-0.5,0.5],\qquad
    x_0^i=[u_0^i,v_0^i]^\top.$$

    Stacking all $x_0^i$ gives a batch of initial states.
    """)
    return


@app.cell
def _(torch):
    def sample_ic(n_traj, device):
        u0 = 0.5 * (2 * torch.rand(n_traj, device=device) - 1)
        v0 = 0.5 * (2 * torch.rand(n_traj, device=device) - 1)
        return torch.stack([u0, v0], dim=-1)

    return (sample_ic,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.2 Reference dynamics $f(t,x)$

    The data-generating physical dynamics is

    $$x=[u,v]^\top,\quad \dot{u}=v,\quad \dot{v}=-\beta v-\frac{g}{\ell}\sin(u).$$

    This defines the ground-truth trajectories used to create observations.
    """)
    return


@app.cell
def _(torch):
    def pendulum_rhs(x, beta, ell, g):
        u, v = x[..., 0], x[..., 1]
        du = v
        dv = -beta * v - (g / ell) * torch.sin(u)
        return torch.stack([du, dv], dim=-1)

    return (pendulum_rhs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.3 One RK4 step for autonomous ODEs

    For step size $h_k=t_{k+1}-t_k$, RK4 uses

    $$k_1=f(x_k),\quad
    k_2=f\!\left(x_k+\frac{h_k}{2}k_1\right),\quad
    k_3=f\!\left(x_k+\frac{h_k}{2}k_2\right),\quad
    k_4=f(x_k+h_k k_3),$$

    then updates

    $$x_{k+1}=x_k+\frac{h_k}{6}(k_1+2k_2+2k_3+k_4).$$
    """)
    return


@app.function
def rk4_step_autonomous(f, x_k, h_k):
    k1 = f(x_k)
    k2 = f(x_k + 0.5 * h_k * k1)
    k3 = f(x_k + 0.5 * h_k * k2)
    k4 = f(x_k + h_k * k3)
    return x_k + (h_k / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.4 RK4 rollout on the full grid

    Repeating the one-step update for $k=0,\dots,N-1$ produces

    $$X=[x_0, x_1, \dots, x_N].$$

    These stored forward states are reused by the custom adjoint backward pass.
    """)
    return


@app.cell
def _(torch):
    def rk4_integrate_autonomous(f, x0, t_grid):
        x_k = x0
        states = [x_k]
        for k in range(t_grid.shape[0] - 1):
            h_k = t_grid[k + 1] - t_grid[k]
            x_k = rk4_step_autonomous(f, x_k, h_k)
            states.append(x_k)
        return torch.stack(states, dim=0)

    return (rk4_integrate_autonomous,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Adjoint Solver Components

    We now define the neural vector field and the augmented adjoint dynamics used in `backward`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.1 Neural vector field module $f_\theta$

    `ODEFunc` parameterizes

    $$f_\theta:\mathbb{R}^n\to\mathbb{R}^n,\qquad \dot{x}=f_\theta(x).$$

    Here $n=2$ (angle and angular velocity).
    """)
    return


@app.cell
def _(nn):
    class ODEFunc(nn.Module):
        def __init__(self, hidden_dim=128, use_he_init=True, device="cpu"):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),
            ).to(device)

            if use_he_init:
                for module in self.net:
                    if isinstance(module, nn.Linear):
                        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                        if module.bias is not None:
                            nn.init.constant_(module.bias, 0.0)

        def forward(self, x):
            return self.net(x)

    return (ODEFunc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.2 Augmented adjoint dynamics $(x,a,g)$

    Let

    $$J_x(x,\theta)=\nabla_x f_\theta(x)\in\mathbb{R}^{n\times n},\qquad
    J_\theta(x,\theta)=\nabla_\theta f_\theta(x).$$

    For backward integration, we evaluate

    $$\dot{x}=f_\theta(x),\qquad
    \dot{a}=-J_x(x,\theta)^\top a,\qquad
    \dot{g}=-J_\theta(x,\theta)^\top a.$$

    In code we do not explicitly form $J_x$ or $J_\theta$. Instead, we compute
    vector-Jacobian products:

    $$\frac{\partial\langle -a,f_\theta(x)\rangle}{\partial x}
    =-J_x^\top a,\qquad
    \frac{\partial\langle -a,f_\theta(x)\rangle}{\partial \theta}
    =-J_\theta^\top a.$$

    In code, `torch.autograd.grad` computes these sensitivity products directly
    (without manually building full Jacobian matrices), using `grad_outputs=-a`.

    This function returns one augmented derivative tuple
    $(\dot{x},\dot{a},\dot{g})$.
    """)
    return


@app.cell
def _(torch):
    def augmented_dynamics(func, x, a, params):
        with torch.enable_grad():
            x_req = x.detach().requires_grad_(True)
            f_val = func(x_req)
            grads = torch.autograd.grad(
                outputs=f_val,
                inputs=(x_req, *params),
                grad_outputs=-a,
                allow_unused=True,
                retain_graph=False,
                create_graph=False,
            )

        dadt = grads[0] if grads[0] is not None else torch.zeros_like(x_req)
        dgdt = []
        for grad_param, param in zip(grads[1:], params):
            dgdt.append(grad_param if grad_param is not None else torch.zeros_like(param))

        return f_val.detach(), dadt.detach(), [item.detach() for item in dgdt]

    return (augmented_dynamics,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.3 RK4 step for the augmented system

    We apply RK4 simultaneously to all augmented states:

    $$(x,a,g)_{k+1}=(x,a,g)_k+\frac{h_k}{6}
    \Big(K_1+2K_2+2K_3+K_4\Big),$$

    where each $K_j$ contains the derivatives for $(x,a,g)$ at the corresponding RK4 stage.

    In backward time, each step size is

    $$h_k=t_{k-1}-t_k<0,$$

    so integrating with RK4 naturally moves from $t_k$ to $t_{k-1}$.

    The per-interval parameter-gradient contribution is the RK4 approximation of

    $$\Delta g_k\approx\int_{t_k}^{t_{k-1}}-(\nabla_\theta f_\theta)^\top a\,dt,$$

    which the code adds to `grad_params`.
    """)
    return


@app.cell
def _(augmented_dynamics):
    def rk4_augmented_step(func, x, a, h_k, params):
        k1x, k1a, k1g = augmented_dynamics(func, x, a, params)
        k2x, k2a, k2g = augmented_dynamics(func, x + 0.5 * h_k * k1x, a + 0.5 * h_k * k1a, params)
        k3x, k3a, k3g = augmented_dynamics(func, x + 0.5 * h_k * k2x, a + 0.5 * h_k * k2a, params)
        k4x, k4a, k4g = augmented_dynamics(func, x + h_k * k3x, a + h_k * k3a, params)

        x_next = x + (h_k / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
        a_next = a + (h_k / 6.0) * (k1a + 2 * k2a + 2 * k3a + k4a)

        g_next = []
        for j in range(len(params)):
            g_next.append((h_k / 6.0) * (k1g[j] + 2 * k2g[j] + 2 * k3g[j] + k4g[j]))

        return x_next, a_next, g_next

    return (rk4_augmented_step,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.4 Custom autograd operator with adjoint backward pass

    `forward`:

    - integrates the forward ODE trajectory,
    - stores $(t_k, x_k)$ for replay in reverse time.

    `backward`:

    1. starts from terminal sensitivity $a_N=\partial\mathcal{L}/\partial x_N$,
    2. integrates the augmented dynamics from $t_N$ to $t_0$,
    3. accumulates parameter gradient contributions into $g$,
    4. applies discrete jump corrections:

    $$a_{k-1}=a_{k-1}^{\text{int}} + \nabla_{x_{k-1}}\ell_{k-1}.$$

    Exact mapping to code variables:

    - `grad_y[k]` is $\partial\mathcal{L}/\partial x_k$ from PyTorch's chain rule.
    - `a = grad_y[-1]` initializes $a_N$.
    - one backward interval computes $(a_{k-1}^{\text{int}}, \Delta g_k)$ via `rk4_augmented_step`.
    - `grad_params += g_step` accumulates $\sum_k \Delta g_k \approx g(t_0)$.
    - `a = a + grad_y[k-1]` applies the jump at $t_{k-1}$.

    So the returned parameter gradients are exactly the adjoint estimate of
    $\nabla_\theta\mathcal{L}$, and `grad_x0` is $\partial\mathcal{L}/\partial x_0$.
    """)
    return


@app.cell
def _(rk4_augmented_step, rk4_integrate_autonomous, torch):
    class ODEAdjointRK4(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x0, t_grid, func, *params):
            with torch.no_grad():
                y = rk4_integrate_autonomous(lambda x: func(x), x0, t_grid)
            ctx.func = func
            ctx.save_for_backward(t_grid, y, *params)
            return y

        @staticmethod
        def backward(ctx, grad_y):
            t_grid, y, *params = ctx.saved_tensors
            func = ctx.func

            grad_params = [torch.zeros_like(param) for param in params]
            a = grad_y[-1]

            for k in range(t_grid.shape[0] - 1, 0, -1):
                h_k = t_grid[k - 1] - t_grid[k]
                x_k = y[k]
                _, a, g_step = rk4_augmented_step(func, x_k, a, h_k, params)

                for j in range(len(grad_params)):
                    grad_params[j] = grad_params[j] + g_step[j]

                a = a + grad_y[k - 1]

            grad_x0 = a
            return grad_x0, None, None, *grad_params

    return (ODEAdjointRK4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.5 Neural ODE wrapper using the custom adjoint

    `NeuralODEAdjoint` wraps the custom operator behind the standard NODE API:

    $$X = \mathrm{ODESolve}(f_\theta, x_0, \{t_k\}_{k=0}^{N}).$$

    So training code can stay identical to standard PyTorch modules.
    """)
    return


@app.cell
def _(ODEAdjointRK4, ODEFunc, nn):
    class NeuralODEAdjoint(nn.Module):
        def __init__(self, hidden_dim=128, use_he_init=True, device="cpu"):
            super().__init__()
            self.func = ODEFunc(hidden_dim=hidden_dim, use_he_init=use_he_init, device=device)

        def forward(self, x0, t_grid):
            params = tuple(self.func.parameters())
            return ODEAdjointRK4.apply(x0, t_grid, self.func, *params)

    return (NeuralODEAdjoint,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Data, Loss Terms, and Training

    This section builds $\mathcal{D}$, computes $\mathcal{L}$, and optimizes $\theta$ using
    gradients supplied by the custom adjoint `backward`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.1 Dataset builder $\mathcal{D}$

    We generate synthetic supervised trajectories:

    $$x_{k}^i=\Phi_{t_0\to t_k}(x_0^i),\qquad
    \hat{x}_{k}^i=x_k^i+\eta_k^i.$$

    The resulting dataset is

    $$\mathcal{D}=\Big\{\big(\{(t_k,\hat{x}_k^i)\}_{k=0}^{N},x_0^i\big)\Big\}_{i=1}^{M}.$$
    """)
    return


@app.cell
def _(pendulum_rhs, rk4_integrate_autonomous, sample_ic, torch):
    def make_observation_dataset(g_true, beta_true, ell_true, t_final, n_steps, n_traj, noise_std, device):
        t_grid = torch.linspace(0.0, t_final, n_steps, device=device)
        x0_batch = sample_ic(n_traj, device)

        with torch.no_grad():
            x_true = rk4_integrate_autonomous(
                lambda x: pendulum_rhs(x, beta_true, ell_true, g_true),
                x0_batch,
                t_grid,
            )

        x_obs = (x_true + noise_std * torch.randn_like(x_true)).detach()
        return t_grid, x0_batch, x_true, x_obs

    return (make_observation_dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.2 Observation normalization

    We normalize with empirical observation statistics:

    $$\tilde{x}=\frac{x-\mu_{\hat{x}}}{\sigma_{\hat{x}}+\varepsilon}.$$

    Inference outputs are later rescaled to compare against physical-space trajectories.
    """)
    return


@app.function
def compute_normalization_stats(x0_batch, x_obs):
    x_obs_mean = x_obs.mean(dim=[0, 1], keepdim=True)
    x_obs_std = x_obs.std(dim=[0, 1], keepdim=True) + 1e-6
    x0_train = (x0_batch - x_obs_mean[0]) / x_obs_std[0]
    return x_obs_mean, x_obs_std, x0_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.3 Optional regularization term

    Data term:

    $$\mathcal{L}_{\text{data}}=
    \frac{1}{MN}\sum_{i=1}^M\sum_{k=0}^{N}\|x_k^i-\hat{x}_k^i\|_2^2.$$

    Regularizer:

    $$R(\theta)=\frac{1}{\operatorname{mean}\|f_\theta(x)\|_2+\varepsilon},\qquad
    \mathcal{L}=\mathcal{L}_{\text{data}}+\lambda R(\theta).$$

    Toggling regularization changes only the additive $\lambda R(\theta)$ term.
    """)
    return


@app.cell
def _(torch):
    def derivative_regularization_term(vector_field, x_state, reg_strength, use_regularization):
        if not use_regularization:
            zero = torch.tensor(0.0, device=x_state.device)
            nan_value = torch.tensor(float("nan"), device=x_state.device)
            return zero, nan_value

        mean_norm = torch.mean(torch.norm(vector_field(x_state), dim=-1))
        reg_term = reg_strength * (1.0 / (mean_norm + 1e-6))
        return reg_term, mean_norm

    return (derivative_regularization_term,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.4 Training routine

    Each epoch performs:

    1. forward solve for all trajectories,
    2. loss evaluation,
    3. custom-adjoint backward pass,
    4. optimizer/scheduler updates.

    Parameter update:

    $$\theta \leftarrow \theta - \alpha\,\nabla_\theta\mathcal{L}(\theta),$$

    with optional weight decay and gradient clipping for stability.

    Practical interpretation:
    the adjoint method reduces memory usage by recomputing sensitivity dynamics
    backward in time instead of storing all intermediate solver gradients.

    In this notebook, `loss.backward()` triggers the custom adjoint `backward`,
    which computes the parameter gradient through the sequence:

    $$\text{forward solve} \;\rightarrow\; \text{adjoint backward solve}
    \;\rightarrow\; \nabla_\theta\mathcal{L}.$$
    """)
    return


@app.cell
def _(
    NeuralODEAdjoint,
    derivative_regularization_term,
    make_observation_dataset,
    mo,
    nn,
    np,
    torch,
):
    @mo.persistent_cache
    def train_model(
        seed,
        g_true,
        beta_true,
        ell_true,
        t_final,
        n_steps,
        n_traj,
        noise_std,
        hidden_dim,
        use_he_init,
        epochs,
        lr,
        use_weight_decay,
        weight_decay,
        use_scheduler,
        scheduler_factor,
        scheduler_patience,
        use_grad_clip,
        clip_norm,
        use_regularization,
        reg_strength,
        print_every,
        device_type,
    ):
        torch.manual_seed(seed)
        device = torch.device(device_type)

        t_grid, x0_batch, x_true, x_obs = make_observation_dataset(
            g_true=g_true,
            beta_true=beta_true,
            ell_true=ell_true,
            t_final=t_final,
            n_steps=n_steps,
            n_traj=n_traj,
            noise_std=noise_std,
            device=device,
        )

        x_obs_mean, x_obs_std, x0_train = compute_normalization_stats(
            x0_batch=x0_batch,
            x_obs=x_obs,
        )

        model = NeuralODEAdjoint(hidden_dim=hidden_dim, use_he_init=use_he_init, device=device).to(device)

        wd = weight_decay if use_weight_decay else 0.0
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        scheduler = None
        if use_scheduler:
            try:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=scheduler_factor,
                    patience=scheduler_patience,
                    verbose=True,
                )
            except TypeError:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=scheduler_factor,
                    patience=scheduler_patience,
                )

        mse = nn.MSELoss()
        total_history = []
        mse_history = []
        reg_history = []
        deriv_history = []
        lr_history = []

        for epoch in range(epochs):
            optimizer.zero_grad()

            x_sim = model(x0_train, t_grid)
            x_sim_rescaled = x_sim * x_obs_std + x_obs_mean
            mse_loss = mse(x_sim_rescaled, x_obs)

            reg_term, deriv_norm = derivative_regularization_term(
                vector_field=model.func.net,
                x_state=x_sim,
                reg_strength=reg_strength,
                use_regularization=use_regularization,
            )

            loss = mse_loss + reg_term
            if not torch.isfinite(loss):
                print(f"[{epoch:04d}] non-finite loss; stopping early")
                break

            loss.backward()

            if use_grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

            optimizer.step()

            if scheduler is not None:
                scheduler.step(mse_loss.detach())

            current_lr = optimizer.param_groups[0]["lr"]
            total_history.append(float(loss.item()))
            mse_history.append(float(mse_loss.item()))
            reg_history.append(float(reg_term.item()))
            deriv_history.append(float(deriv_norm.item()) if use_regularization else np.nan)
            lr_history.append(float(current_lr))

            if epoch % print_every == 0:
                print(
                    f"[{epoch:04d}] total={loss.item():.6f} "
                    f"mse={mse_loss.item():.6f} reg={reg_term.item():.6f} "
                    f"lr={current_lr:.6f}"
                )

        with torch.no_grad():
            x_fit = model(x0_train, t_grid)
            x_fit = x_fit * x_obs_std + x_obs_mean
            final_clean_mse = mse(x_fit, x_true).item()

        return {
            "t_grid": t_grid.detach().cpu().numpy(),
            "x_true": x_true.detach().cpu().numpy(),
            "x_obs": x_obs.detach().cpu().numpy(),
            "x_fit": x_fit.detach().cpu().numpy(),
            "total_history": np.asarray(total_history),
            "mse_history": np.asarray(mse_history),
            "reg_history": np.asarray(reg_history),
            "deriv_history": np.asarray(deriv_history),
            "lr_history": np.asarray(lr_history),
            "final_clean_mse": float(final_clean_mse),
        }

    return (train_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a id="control-panel"></a>
    ## 4. Control Panel

    Defaults mirror the adjoint script setup: normalization, scheduler, clipping,
    weight decay, and regularization start enabled.

    For a simpler baseline, disable regularization and scheduler first, then re-enable progressively.
    Use $N$ and $M$ to trade computational cost vs. trajectory fidelity.
    """)
    return


@app.cell(hide_code=True)
def _(mo, torch):
    seed_num = mo.ui.number(value=0, label="Seed")

    g_slider = mo.ui.slider(5.0, 15.0, value=9.81, step=0.01, label="Gravity g")
    beta_slider = mo.ui.slider(0.05, 1.0, value=0.25, step=0.01, label="Damping beta")
    ell_slider = mo.ui.slider(0.3, 2.0, value=0.9, step=0.01, label="Length ell")

    t_final_slider = mo.ui.slider(1.0, 10.0, value=5.0, step=0.5, label="Final time T")
    n_steps_slider = mo.ui.slider(50, 500, value=200, step=10, label="Time steps N")
    n_traj_slider = mo.ui.slider(4, 128, value=32, step=4, label="Trajectories M")
    noise_slider = mo.ui.slider(0.0, 0.1, value=0.01, step=0.005, label="Observation noise")

    hidden_dim_slider = mo.ui.slider(16, 256, value=128, step=16, label="Hidden dimension")
    use_he_init = mo.ui.checkbox(label="Use He initialization", value=True)

    epochs_dropdown = mo.ui.dropdown(
        {"500": 500, "1000 (script default)": 1000, "2000": 2000},
        value="1000 (script default)",
        label="Epochs",
    )
    lr_dropdown = mo.ui.dropdown(
        {"5e-2": 5e-2, "1e-2 (script default)": 1e-2, "5e-3": 5e-3, "1e-3": 1e-3},
        value="1e-2 (script default)",
        label="Learning rate",
    )
    print_every_slider = mo.ui.slider(10, 500, value=100, step=10, label="Print every")

    use_regularization = mo.ui.checkbox(label="Use derivative regularization", value=True)
    reg_strength_slider = mo.ui.slider(0.0, 0.2, value=0.01, step=0.001, label="Regularization strength")

    use_weight_decay = mo.ui.checkbox(label="Use weight decay", value=True)
    weight_decay_dropdown = mo.ui.dropdown(
        {"1e-5": 1e-5, "1e-4 (script default)": 1e-4, "1e-3": 1e-3},
        value="1e-4 (script default)",
        label="Weight decay",
    )

    use_scheduler = mo.ui.checkbox(label="Use ReduceLROnPlateau", value=True)
    scheduler_factor_slider = mo.ui.slider(0.1, 0.9, value=0.5, step=0.1, label="Scheduler factor")
    scheduler_patience_slider = mo.ui.slider(10, 300, value=100, step=10, label="Scheduler patience")

    use_grad_clip = mo.ui.checkbox(label="Use gradient clipping", value=True)
    clip_norm_slider = mo.ui.slider(0.1, 5.0, value=1.0, step=0.1, label="Clip norm")

    plot_count_slider = mo.ui.slider(1, 8, value=3, step=1, label="Trajectories to plot")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_button = mo.ui.run_button(label="Train Torch Adjoint NODE")

    panel = mo.vstack(
        [
            train_button,
            mo.md(f"**Device:** `{device}`"),
            mo.md("### Physics"),
            g_slider,
            beta_slider,
            ell_slider,
            mo.md("### Data Generation"),
            seed_num,
            t_final_slider,
            n_steps_slider,
            n_traj_slider,
            noise_slider,
            mo.md("### Model"),
            hidden_dim_slider,
            use_he_init,
            mo.md("### Training"),
            epochs_dropdown,
            lr_dropdown,
            print_every_slider,
            mo.md("### Regularization"),
            use_regularization,
            reg_strength_slider,
            mo.md("### Optimization Stabilization"),
            use_weight_decay,
            weight_decay_dropdown,
            use_scheduler,
            scheduler_factor_slider,
            scheduler_patience_slider,
            use_grad_clip,
            clip_norm_slider,
            mo.md("### Visualization"),
            plot_count_slider,
        ]
    )

    panel
    return (
        beta_slider,
        clip_norm_slider,
        device,
        ell_slider,
        epochs_dropdown,
        g_slider,
        hidden_dim_slider,
        lr_dropdown,
        n_steps_slider,
        n_traj_slider,
        noise_slider,
        plot_count_slider,
        print_every_slider,
        reg_strength_slider,
        scheduler_factor_slider,
        scheduler_patience_slider,
        seed_num,
        t_final_slider,
        train_button,
        use_grad_clip,
        use_he_init,
        use_regularization,
        use_scheduler,
        use_weight_decay,
        weight_decay_dropdown,
    )


@app.cell(hide_code=True)
def _(
    beta_slider,
    clip_norm_slider,
    device,
    ell_slider,
    epochs_dropdown,
    g_slider,
    hidden_dim_slider,
    lr_dropdown,
    mo,
    n_steps_slider,
    n_traj_slider,
    noise_slider,
    print_every_slider,
    reg_strength_slider,
    scheduler_factor_slider,
    scheduler_patience_slider,
    seed_num,
    t_final_slider,
    train_button,
    train_model,
    use_grad_clip,
    use_he_init,
    use_regularization,
    use_scheduler,
    use_weight_decay,
    weight_decay_dropdown,
):
    mo.stop(not train_button.value, mo.md("_Click **Train Torch Adjoint NODE** to start training._"))

    results = train_model(
        seed=int(seed_num.value),
        g_true=float(g_slider.value),
        beta_true=float(beta_slider.value),
        ell_true=float(ell_slider.value),
        t_final=float(t_final_slider.value),
        n_steps=int(n_steps_slider.value),
        n_traj=int(n_traj_slider.value),
        noise_std=float(noise_slider.value),
        hidden_dim=int(hidden_dim_slider.value),
        use_he_init=bool(use_he_init.value),
        epochs=int(epochs_dropdown.value),
        lr=float(lr_dropdown.value),
        use_weight_decay=bool(use_weight_decay.value),
        weight_decay=float(weight_decay_dropdown.value),
        use_scheduler=bool(use_scheduler.value),
        scheduler_factor=float(scheduler_factor_slider.value),
        scheduler_patience=int(scheduler_patience_slider.value),
        use_grad_clip=bool(use_grad_clip.value),
        clip_norm=float(clip_norm_slider.value),
        use_regularization=bool(use_regularization.value),
        reg_strength=float(reg_strength_slider.value),
        print_every=int(print_every_slider.value),
        device_type=device.type,
    )
    return (results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 5. Results
    """)
    return


@app.cell(hide_code=True)
def _(mo, results, use_regularization):
    n_epochs = int(results["total_history"].shape[0])
    final_total = float(results["total_history"][-1]) if n_epochs > 0 else float("nan")
    final_mse = float(results["mse_history"][-1]) if n_epochs > 0 else float("nan")
    final_reg = float(results["reg_history"][-1]) if n_epochs > 0 else 0.0
    final_clean_mse = float(results["final_clean_mse"])
    reg_mode = "on" if bool(use_regularization.value) else "off"

    mo.md(
        f"""
        **Training summary**

        - Epochs completed: `{n_epochs}`
        - Final total loss: `{final_total:.6e}`
        - Final data MSE loss: `{final_mse:.6e}`
        - Final regularization term: `{final_reg:.6e}` (regularization {reg_mode})
        - Final clean MSE against noise-free trajectories: `{final_clean_mse:.6e}`
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.1 Trajectory comparison

    For selected trajectories, the plots show:

    - $u_k$ and $v_k$ reference vs prediction,
    - component errors
      $e_u(t_k)=u_k^{\text{pred}}-\hat{u}_k$ and
      $e_v(t_k)=v_k^{\text{pred}}-\hat{v}_k$.
    """)
    return


@app.cell
def _(np, plt):
    def plot_trajectory_comparison(results, n_plot):
        t = results["t_grid"]
        x_true = results["x_true"]
        x_fit = results["x_fit"]

        n_plot = min(int(n_plot), x_true.shape[1])
        idx = np.arange(n_plot)

        fig, axes = plt.subplots(3, n_plot, figsize=(5 * n_plot, 11), sharex=True, squeeze=False)
        for i, k in enumerate(idx):
            axes[0, i].plot(t, x_true[:, k, 0], "k-", label="True angle")
            axes[0, i].plot(t, x_fit[:, k, 0], "r--", label="Adjoint NODE angle")
            axes[0, i].set_ylabel("u(t)")
            axes[0, i].set_title(f"Trajectory {k + 1}: angle")
            axes[0, i].grid(True, alpha=0.3)
            axes[0, i].legend()

            axes[1, i].plot(t, x_true[:, k, 1], "k-", label="True velocity")
            axes[1, i].plot(t, x_fit[:, k, 1], "r--", label="Adjoint NODE velocity")
            axes[1, i].set_ylabel("v(t)")
            axes[1, i].set_title(f"Trajectory {k + 1}: velocity")
            axes[1, i].grid(True, alpha=0.3)
            axes[1, i].legend()

            err_u = x_fit[:, k, 0] - x_true[:, k, 0]
            err_v = x_fit[:, k, 1] - x_true[:, k, 1]
            axes[2, i].plot(t, err_u, "m-", label="u error")
            axes[2, i].plot(t, err_v, "g--", label="v error")
            axes[2, i].axhline(0.0, color="k", linewidth=1.0, alpha=0.5)
            axes[2, i].set_xlabel("Time")
            axes[2, i].set_ylabel("Error")
            axes[2, i].set_title(f"Trajectory {k + 1}: state error")
            axes[2, i].grid(True, alpha=0.3)
            axes[2, i].legend()

        plt.tight_layout()
        return fig

    return (plot_trajectory_comparison,)


@app.cell(hide_code=True)
def _(plot_count_slider, plot_trajectory_comparison, results):
    plot_trajectory_comparison(results, plot_count_slider.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.2 Optimization history

    We monitor the optimization path across epochs:

    - $\mathcal{L}$ (total),
    - $\mathcal{L}_{\text{data}}$,
    - $\lambda R(\theta)$,
    - learning rate schedule.

    The log-scale y-axis reveals both early transients and late fine-tuning.
    """)
    return


@app.cell
def _(np, plt):
    def plot_loss_curves(results):
        total_hist = results["total_history"]
        mse_hist = results["mse_history"]
        reg_hist = results["reg_history"]
        lr_hist = results["lr_history"]

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

        epochs = np.arange(1, len(total_hist) + 1)
        axes[0].plot(epochs, total_hist, label="Total loss", color="tab:blue")
        axes[0].plot(epochs, mse_hist, label="MSE loss", color="tab:orange")
        axes[0].plot(epochs, reg_hist, label="Reg term", color="tab:green")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss curves")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(epochs, lr_hist, color="tab:red")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Learning rate")
        axes[1].set_title("Learning rate schedule")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    return (plot_loss_curves,)


@app.cell(hide_code=True)
def _(plot_loss_curves, results):
    plot_loss_curves(results)
    return


if __name__ == "__main__":
    app.run()
