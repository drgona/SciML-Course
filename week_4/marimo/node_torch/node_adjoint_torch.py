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
    # Neural ODE Training with a Custom Adjoint (PyTorch)

    This notebook converts `week_4/scripts/torch/NODE_adjoint_example.py` into an interactive marimo app.

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
    ## 1. Discrete RK4 Dynamics and Adjoint Gradient

    True and learned systems:

    $$x(t)=\begin{bmatrix}u(t)\\v(t)\end{bmatrix},\qquad
    \dot{u}=v,\qquad
    \dot{v}=-\beta v-\frac{g}{\ell}\sin(u),\qquad
    \dot{x}=f_\theta(x).$$

    RK4 one-step map:

    $$x_k=\Phi_\theta(x_{k-1}).$$

    Trajectory objective:

    $$\mathcal{L}=\sum_{k=0}^{N}\ell(x_k),\qquad
    \ell(x_k)=\|x_k^{\mathrm{sim}}-x_k^{\mathrm{obs}}\|_2^2.$$

    Discrete adjoint recursion (backward in time):

    $$a_{k-1}=a_k\frac{\partial\Phi_\theta}{\partial x_{k-1}} + \frac{\partial\ell_{k-1}}{\partial x_{k-1}},
    \qquad a_N=\frac{\partial\ell_N}{\partial x_N}.$$

    Parameter gradient accumulation:

    $$\nabla_\theta \mathcal{L} = \sum_{k=1}^{N} a_k\frac{\partial\Phi_\theta}{\partial\theta}.$$

    The notebook implements this with a custom `torch.autograd.Function` that:
    1. runs forward RK4 in `forward`,
    2. replays a backward-time augmented system in `backward`.
    """)
    return


@app.cell(hide_code=True)
def _(torch):
    def sample_ic(n_traj, device):
        u0 = 0.5 * (2 * torch.rand(n_traj, device=device) - 1)
        v0 = 0.5 * (2 * torch.rand(n_traj, device=device) - 1)
        return torch.stack([u0, v0], dim=-1)

    def pendulum_rhs(x, beta, ell, g):
        u, v = x[..., 0], x[..., 1]
        du = v
        dv = -beta * v - (g / ell) * torch.sin(u)
        return torch.stack([du, dv], dim=-1)

    def rk4_step_autonomous(f, x, dt):
        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def rk4_integrate_autonomous(f, x0, t):
        x = x0
        xs = [x]
        for k in range(t.shape[0] - 1):
            dt = t[k + 1] - t[k]
            x = rk4_step_autonomous(f, x, dt)
            xs.append(x)
        return torch.stack(xs, dim=0)

    return pendulum_rhs, rk4_integrate_autonomous, sample_ic


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.1 Code: Custom Adjoint Operator

    The next code cell builds:

    $$\frac{da}{dt}=-a^\top\frac{\partial f_\theta}{\partial x},\qquad
    \frac{d}{dt}\nabla_\theta\mathcal{L}=-a^\top\frac{\partial f_\theta}{\partial\theta},$$

    and integrates this augmented backward system with RK4.
    """)
    return


@app.cell(hide_code=True)
def _(nn, rk4_integrate_autonomous, torch):
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

    def _augmented_dynamics(func, x, a, params):
        with torch.enable_grad():
            x_req = x.detach().requires_grad_(True)
            f = func(x_req)
            grads = torch.autograd.grad(
                outputs=f,
                inputs=(x_req, *params),
                grad_outputs=-a,
                allow_unused=True,
                retain_graph=False,
                create_graph=False,
            )

        dadt = grads[0] if grads[0] is not None else torch.zeros_like(x_req)
        dpdt = []
        for g, p in zip(grads[1:], params):
            dpdt.append(g if g is not None else torch.zeros_like(p))
        return f.detach(), dadt.detach(), [item.detach() for item in dpdt]

    def _rk4_augmented_step(func, x, a, dt, params):
        k1x, k1a, k1p = _augmented_dynamics(func, x, a, params)
        k2x, k2a, k2p = _augmented_dynamics(func, x + 0.5 * dt * k1x, a + 0.5 * dt * k1a, params)
        k3x, k3a, k3p = _augmented_dynamics(func, x + 0.5 * dt * k2x, a + 0.5 * dt * k2a, params)
        k4x, k4a, k4p = _augmented_dynamics(func, x + dt * k3x, a + dt * k3a, params)

        x_new = x + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
        a_new = a + (dt / 6.0) * (k1a + 2 * k2a + 2 * k3a + k4a)
        grad_params = []
        for j in range(len(params)):
            grad_params.append((dt / 6.0) * (k1p[j] + 2 * k2p[j] + 2 * k3p[j] + k4p[j]))
        return x_new, a_new, grad_params

    class ODEAdjointRK4(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x0, t, func, *params):
            with torch.no_grad():
                y = rk4_integrate_autonomous(lambda x: func(x), x0, t)
            ctx.func = func
            ctx.save_for_backward(t, y, *params)
            return y

        @staticmethod
        def backward(ctx, grad_y):
            t, y, *params = ctx.saved_tensors
            func = ctx.func

            grad_params = [torch.zeros_like(p) for p in params]
            a = grad_y[-1]

            for i in range(t.shape[0] - 1, 0, -1):
                dt = t[i - 1] - t[i]
                x_i = y[i]
                _, a, gp = _rk4_augmented_step(func, x_i, a, dt, params)
                for j in range(len(grad_params)):
                    grad_params[j] = grad_params[j] + gp[j]
                a = a + grad_y[i - 1]

            grad_x0 = a
            return grad_x0, None, None, *grad_params

    class NeuralODEAdjoint(nn.Module):
        def __init__(self, hidden_dim=128, use_he_init=True, device="cpu"):
            super().__init__()
            self.func = ODEFunc(hidden_dim=hidden_dim, use_he_init=use_he_init, device=device)

        def forward(self, x0, t):
            params = tuple(self.func.parameters())
            return ODEAdjointRK4.apply(x0, t, self.func, *params)

    return (NeuralODEAdjoint,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.2 Code: Training Objective and Optimization

    We optimize:

    $$\mathcal{L}=\mathcal{L}_{\mathrm{data}}+\lambda\mathcal{R},\qquad
    \mathcal{R}=\frac{1}{\operatorname{mean}\|f_\theta(x)\|_2+\varepsilon},$$

    with Adam, optional weight decay, clipping, and plateau LR scheduling.
    """)
    return


@app.cell(hide_code=True)
def _(
    NeuralODEAdjoint,
    mo,
    nn,
    np,
    pendulum_rhs,
    rk4_integrate_autonomous,
    sample_ic,
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

        t_grid = torch.linspace(0.0, t_final, n_steps, device=device)
        x0_batch = sample_ic(n_traj, device)

        with torch.no_grad():
            x_true = rk4_integrate_autonomous(
                lambda x: pendulum_rhs(x, beta_true, ell_true, g_true),
                x0_batch,
                t_grid,
            )

        x_obs = (x_true + noise_std * torch.randn_like(x_true)).detach()
        x_obs_mean = x_obs.mean(dim=[0, 1], keepdim=True)
        x_obs_std = x_obs.std(dim=[0, 1], keepdim=True) + 1e-6
        x0_batch_normalized = (x0_batch - x_obs_mean[0]) / x_obs_std[0]

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

            x_sim = model(x0_batch_normalized, t_grid)
            x_sim_rescaled = x_sim * x_obs_std + x_obs_mean
            mse_loss = mse(x_sim_rescaled, x_obs)

            if use_regularization:
                deriv_norm = torch.mean(torch.norm(model.func.net(x_sim), dim=-1))
                reg_term = reg_strength * (1.0 / (deriv_norm + 1e-6))
            else:
                deriv_norm = torch.tensor(float("nan"), device=device)
                reg_term = torch.tensor(0.0, device=device)

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
            x_fit = model(x0_batch_normalized, t_grid)
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

    Defaults mirror the script-level adjoint setup: normalized data, scheduler, clipping,
    weight decay, and regularization are all available with practical starting values.

    For a simpler baseline, disable regularization and scheduler first, then re-enable progressively.
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
def _(np, plot_count_slider, plt, results):
    def _():
        def _():
            def _():
                t = results["t_grid"]
                x_true = results["x_true"]
                x_fit = results["x_fit"]

                n_plot = min(int(plot_count_slider.value), x_true.shape[1])
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
            return _()
        return _()


    _()
    return


@app.cell(hide_code=True)
def _(np, plt, results):
    def _():
        def _():
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
        return _()


    _()
    return


if __name__ == "__main__":
    app.run()
