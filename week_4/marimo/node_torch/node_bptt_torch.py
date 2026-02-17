# /// script
# [tool.marimo.runtime]
# auto_instantiate = false
# on_cell_change = "lazy"
# ///

import marimo

__generated_with = "0.19.9"
app = marimo.App(app_title="Neural ODE BPTT (Torch)", auto_download=["html"])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Neural ODE Training with BPTT (PyTorch)

    This notebook turns `week_4/scripts/torch/NODE_BPTT_vanilla.py` and
    `week_4/scripts/torch/NODE_BPTT_tuned.py` into an interactive didactical notebook.

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

    We observe noisy trajectories from a damped pendulum:

    $$x(t)=\begin{bmatrix}u(t)\\v(t)\end{bmatrix},\quad
    \dot{u}=v,\quad
    \dot{v}=-\beta v - \frac{g}{\ell}\sin(u).$$

    We train a Neural ODE model

    $$\frac{dx}{dt}=f_\theta(x),\qquad x(t_0)=x_0,$$

    and compute predictions on a time grid with RK4, which defines

    $$x_{k+1}\approx \Phi_\theta(x_k,h_k),\qquad h_k=t_{k+1}-t_k.$$

    The training objective is trajectory matching with optional regularization:

    $$\mathcal{L}(\theta)
    =\frac{1}{MN}\sum_{i=1}^{M}\sum_{k=0}^{N}\|x_k^i-\hat{x}_k^i\|_2^2+\lambda R(\theta).$$

    Notation used in this notebook:

    - $t$: continuous time, with $t_0$ (start) and $T$ (end).
    - $t_k$: sampled time points on the grid.
    - $x(t)\in\mathbb{R}^n$: state, and $x_k\approx x(t_k)$ on the grid.
    - $x_0$: initial state at $t_0$.
    - $f_\theta$: neural vector field with parameters $\theta\in\mathbb{R}^p$.
    - $M$: number of trajectories, $N$: number of time steps.
    - $\hat{x}_k^i$: observed state for trajectory $i$ at time index $k$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.1 Initial condition sampler $x_0$

    For each trajectory $i\in\{1,\dots,M\}$ we sample

    $$u_0^i \sim \mathcal{U}[-0.5,0.5],\qquad v_0^i \sim \mathcal{U}[-0.5,0.5],\qquad
    x_0^i=[u_0^i,v_0^i]^\top.$$

    Stacking all samples gives an initial-condition batch of shape $(M,2)$.
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
    ### 1.2 True dynamics $f(t,x)$ for data generation

    We generate supervision with the known physical vector field

    $$f(t,x)=
    \begin{bmatrix}
    v\\
    -\beta v-(g/\ell)\sin(u)
    \end{bmatrix}.$$

    This is used only for producing synthetic data $(t_k,\hat{x}_k)$, not for training gradients.
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
    ### 1.3 One RK4 step map $\mathrm{Step}(f,t_k,x_k,h_k)$

    Classical RK4 evaluates four slope stages:

    $$k_1=f(x_k),\quad
    k_2=f\!\left(x_k+\frac{h_k}{2}k_1\right),\quad
    k_3=f\!\left(x_k+\frac{h_k}{2}k_2\right),\quad
    k_4=f(x_k+h_k k_3).$$

    Then the one-step update is

    $$x_{k+1}=x_k+\frac{h_k}{6}(k_1+2k_2+2k_3+k_4).$$
    """)
    return


@app.function
def rk4_step(f, x_k, h_k):
    k1 = f(x_k)
    k2 = f(x_k + 0.5 * h_k * k1)
    k3 = f(x_k + 0.5 * h_k * k2)
    k4 = f(x_k + h_k * k3)
    return x_k + (h_k / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.4 Trajectory integrator on $\{t_k\}_{k=0}^N$

    Starting from $x_0$, we repeatedly apply the one-step map:

    $$x_{k+1}=\mathrm{Step}(f,t_k,x_k,h_k),\qquad k=0,\dots,N-1.$$

    The output is the full trajectory tensor

    $$X = [x_0, x_1, \dots, x_N].$$
    """)
    return


@app.cell
def _(torch):
    def rk4_integrate(f, x0, t_grid):
        x_k = x0
        states = [x_k]
        for k in range(t_grid.shape[0] - 1):
            h_k = t_grid[k + 1] - t_grid[k]
            x_k = rk4_step(f, x_k, h_k)
            states.append(x_k)
        return torch.stack(states, dim=0)

    return (rk4_integrate,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.5 Neural vector field $f_\theta$ and NODE flow

    The model parameterizes the dynamics by a neural network:

    $$\dot{x}=f_\theta(x)=\mathrm{NN}_\theta(x).$$

    A NODE layer maps $x(t_0)\mapsto x(t_1)$ through numerical integration:

    $$x(t_1)=\Phi_{t_0\to t_1}(x_0;\theta),$$

    where $\Phi_{t_0\to t_1}$ is approximated by RK4 over the grid.
    """)
    return


@app.cell
def _(nn, rk4_integrate):
    class NeuralODE(nn.Module):
        def __init__(self, hidden_dim=64, use_he_init=False, device="cpu"):
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

        def forward(self, x0, t_grid):
            return rk4_integrate(lambda x: self.net(x), x0, t_grid)

    return (NeuralODE,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.6 Dataset builder $\mathcal{D}$

    We sample $M$ trajectories and construct observations with additive noise:

    $$\hat{x}^i_k = x^i_k + \eta^i_k.$$

    The dataset can be written as

    $$\mathcal{D}=\Big\{\big(\{(t_k,\hat{x}^i_k)\}_{k=0}^{N},x_0^i\big)\Big\}_{i=1}^{M}.$$
    """)
    return


@app.cell
def _(pendulum_rhs, rk4_integrate, sample_ic, torch):
    def make_observation_dataset(g_true, beta_true, ell_true, t_final, n_steps, n_traj, noise_std, device):
        t_grid = torch.linspace(0.0, t_final, n_steps, device=device)
        x0_batch = sample_ic(n_traj, device)

        with torch.no_grad():
            x_true = rk4_integrate(
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
    ### 1.7 Optional normalization

    To improve conditioning, we optionally normalize states with observation statistics:

    $$\tilde{x} = \frac{x-\mu_{\hat{x}}}{\sigma_{\hat{x}}+\varepsilon}.$$

    Predictions are mapped back to physical units via

    $$x=\tilde{x}(\sigma_{\hat{x}}+\varepsilon)+\mu_{\hat{x}}.$$
    """)
    return


@app.cell
def _(torch):
    def compute_normalization_stats(x0_batch, x_obs, use_data_normalization):
        if use_data_normalization:
            x_obs_mean = x_obs.mean(dim=[0, 1], keepdim=True)
            x_obs_std = x_obs.std(dim=[0, 1], keepdim=True) + 1e-6
            x0_train = (x0_batch - x_obs_mean[0]) / x_obs_std[0]
        else:
            x_obs_mean = torch.zeros((1, 1, 2), device=x_obs.device, dtype=x_obs.dtype)
            x_obs_std = torch.ones((1, 1, 2), device=x_obs.device, dtype=x_obs.dtype)
            x0_train = x0_batch

        return x_obs_mean, x_obs_std, x0_train

    return (compute_normalization_stats,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.8 Optional regularization $\lambda R(\theta)$

    The trajectory-matching objective is

    $$\mathcal{L}_{\text{data}}(\theta)=\frac{1}{MN}\sum_{i=1}^{M}\sum_{k=0}^{N}\|x^i_k-\hat{x}^i_k\|_2^2.$$

    We can add a derivative-magnitude regularizer

    $$R(\theta)=\frac{1}{\operatorname{mean}\|f_\theta(x)\|_2+\varepsilon},\quad
    \mathcal{L}=\mathcal{L}_{\text{data}}+\lambda R(\theta).$$

    Larger $\lambda$ penalizes large average vector-field norms more strongly.
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
    ## 2. BPTT Training Function

    Backpropagation Through Time (BPTT) means:

    1. roll the system forward over all time steps and store intermediate states,
    2. propagate gradients backward through each stored step,
    3. accumulate parameter gradients from every step.

    With discrete dynamics $x_{k+1}=\Phi_\theta(x_k)$ and per-step loss $\ell_k$,
    define $s_k=\partial\mathcal{L}/\partial x_k$.

    Then the backward recursion is

    $$s_k = s_{k+1}\frac{\partial \Phi_\theta}{\partial x_k} + \frac{\partial \ell_k}{\partial x_k},
    \qquad
    \nabla_\theta\mathcal{L}=\sum_k s_{k+1}\frac{\partial \Phi_\theta}{\partial\theta}.$$

    In plain words: each time step contributes to the final gradient, so we must
    differentiate through the whole rollout, not just the last step.

    The training loop below implements the update

    $$\theta \leftarrow \theta-\alpha\nabla_\theta\mathcal{L},$$

    with optional weight decay, gradient clipping, and learning-rate scheduling.
    """)
    return


@app.cell
def _(
    NeuralODE,
    compute_normalization_stats,
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
        use_data_normalization,
        use_regularization,
        reg_strength,
        use_weight_decay,
        weight_decay,
        use_scheduler,
        scheduler_factor,
        scheduler_patience,
        use_grad_clip,
        clip_norm,
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
            use_data_normalization=use_data_normalization,
        )

        model = NeuralODE(hidden_dim=hidden_dim, use_he_init=use_he_init, device=device).to(device)

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
                vector_field=model.net,
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
    ## 3. Control Panel

    **Vanilla default configuration** (matches the basic BPTT script):

    - `Use regularization term` = off
    - `Use data normalization` = off
    - `Use weight decay` = off
    - `Use ReduceLROnPlateau` = off
    - `Use gradient clipping` = off
    - `hidden_dim = 64`, `lr = 5e-2`, `epochs = 501`

    **How to tune from vanilla:**

    1. Toggle `Use regularization term` on and start with $\lambda=0.01$.
    2. If trajectories are too stiff/flat, decrease $\lambda$; if too oscillatory, increase $\lambda$.
    3. Then optionally enable normalization, weight decay, scheduler, and clipping.
    4. If optimization oscillates, lower `lr` (e.g. `1e-2` or `5e-3`) and/or increase epochs.
    5. `N` and `M` control discretization and dataset richness:
       larger `N` improves integration fidelity; larger `M` improves generalization.
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

    hidden_dim_slider = mo.ui.slider(16, 256, value=64, step=16, label="Hidden dimension")
    use_he_init = mo.ui.checkbox(label="Use He initialization", value=False)

    epochs_dropdown = mo.ui.dropdown(
        {"501 (vanilla)": 501, "1000": 1000, "2000": 2000, "3000": 3000},
        value="501 (vanilla)",
        label="Epochs",
    )
    lr_dropdown = mo.ui.dropdown(
        {"5e-2 (vanilla)": 5e-2, "1e-2": 1e-2, "5e-3": 5e-3, "1e-3": 1e-3},
        value="5e-2 (vanilla)",
        label="Learning rate",
    )
    print_every_slider = mo.ui.slider(10, 500, value=100, step=10, label="Print every")

    use_regularization = mo.ui.checkbox(label="Use regularization term", value=False)
    reg_strength_slider = mo.ui.slider(0.0, 0.2, value=0.01, step=0.001, label="Regularization strength lambda")

    use_norm = mo.ui.checkbox(label="Use data normalization", value=False)
    use_weight_decay = mo.ui.checkbox(label="Use weight decay", value=False)
    weight_decay_dropdown = mo.ui.dropdown(
        {"1e-5": 1e-5, "1e-4": 1e-4, "1e-3": 1e-3},
        value="1e-4",
        label="Weight decay",
    )

    use_scheduler = mo.ui.checkbox(label="Use ReduceLROnPlateau", value=False)
    scheduler_factor_slider = mo.ui.slider(0.1, 0.9, value=0.5, step=0.1, label="Scheduler factor")
    scheduler_patience_slider = mo.ui.slider(10, 300, value=100, step=10, label="Scheduler patience")

    use_grad_clip = mo.ui.checkbox(label="Use gradient clipping", value=False)
    clip_norm_slider = mo.ui.slider(0.1, 5.0, value=1.0, step=0.1, label="Clip norm")

    plot_count_slider = mo.ui.slider(1, 8, value=3, step=1, label="Trajectories to plot")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_button = mo.ui.run_button(label="Train Torch BPTT NODE")

    control_panel = mo.vstack(
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
            mo.md("### Optional Stabilization"),
            use_norm,
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

    control_panel
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
        use_norm,
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
    use_norm,
    use_regularization,
    use_scheduler,
    use_weight_decay,
    weight_decay_dropdown,
):
    mo.stop(not train_button.value, mo.md("_Click **Train Torch BPTT NODE** to start training._"))

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
        use_data_normalization=bool(use_norm.value),
        use_regularization=bool(use_regularization.value),
        reg_strength=float(reg_strength_slider.value),
        use_weight_decay=bool(use_weight_decay.value),
        weight_decay=float(weight_decay_dropdown.value),
        use_scheduler=bool(use_scheduler.value),
        scheduler_factor=float(scheduler_factor_slider.value),
        scheduler_patience=int(scheduler_patience_slider.value),
        use_grad_clip=bool(use_grad_clip.value),
        clip_norm=float(clip_norm_slider.value),
        print_every=int(print_every_slider.value),
        device_type=device.type,
    )
    return (results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 4. Results
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
    ### 4.1 Trajectory comparison

    For selected trajectories, we compare:

    - reference state $x_k$,
    - model prediction $x_k^{\text{NODE}}$,
    - component-wise errors
      $e_u(t_k)=u_k^{\text{NODE}}-\hat{u}_k$ and
      $e_v(t_k)=v_k^{\text{NODE}}-\hat{v}_k$.
    """)
    return


@app.cell(hide_code=True)
def _(np, plt):
    def plot_trajectory_comparison(results, n_plot):
        t = results["t_grid"]
        x_true = results["x_true"]
        x_fit = results["x_fit"]

        n_plot = min(int(n_plot), x_true.shape[1])
        idx = np.arange(n_plot)

        fig, axes = plt.subplots(3, n_plot, figsize=(5 * n_plot, 11), sharex=True, squeeze=False)
        for i, k in enumerate(idx):
            axes[0, i].plot(t, x_true[:, k, 0], "b-", label="True angle")
            axes[0, i].plot(t, x_fit[:, k, 0], "r--", label="NODE angle")
            axes[0, i].set_ylabel("u(t)")
            axes[0, i].set_title(f"Trajectory {k + 1}: angle")
            axes[0, i].grid(True, alpha=0.3)
            axes[0, i].legend()

            axes[1, i].plot(t, x_true[:, k, 1], "b-", label="True velocity")
            axes[1, i].plot(t, x_fit[:, k, 1], "r--", label="NODE velocity")
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
    ### 4.2 Optimization history

    We plot the epoch-wise evolution of

    - total objective $\mathcal{L}$,
    - data term $\mathcal{L}_{\text{data}}$,
    - regularization term $\lambda R(\theta)$,
    - learning rate.

    The loss axis is logarithmic to highlight both transient and late-stage behavior.
    """)
    return


@app.cell(hide_code=True)
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
