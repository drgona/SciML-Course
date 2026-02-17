# /// script
# [tool.marimo.runtime]
# auto_instantiate = false
# on_cell_change = "lazy"
# ///

import marimo

__generated_with = "0.19.9"
app = marimo.App(app_title="Neural ODE BPTT (JAX)", auto_download=["html"])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Neural ODE Training with BPTT (JAX)

    This notebook turns `week_4/scripts/jax/NODE_BPTT_vanilla.py` and
    `week_4/scripts/jax/NODE_BPTT_tuned.py` into a didactical marimo notebook.

    **Navigation:** [Jump to Control Panel](#control-panel)
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np

    return jax, jnp, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 1. Problem Formulation

    We model pendulum trajectories with a Neural ODE.

    Reference dynamics:

    $$
    \begin{aligned}
    x(t) &= [u(t), v(t)]^\top \\
    \frac{du}{dt} &= v \\
    \frac{dv}{dt} &= -\beta v - \frac{g}{\ell}\sin(u)
    \end{aligned}
    $$


    Learned dynamics:

    $$
    \frac{dx}{dt} = f_\theta(x), \qquad x(t_0)=x_0.
    $$


    Time-discretized solver update:

    $$
    x_{k+1} \approx \Phi_\theta(x_k, h_k), \qquad h_k=t_{k+1}-t_k.
    $$


    Data model and objective:

    $$
    \hat{x}_k^i = x_k^i + \eta_k^i,
    $$


    $$
    \mathcal{L}(\theta)=\frac{1}{MN}\sum_{i=1}^{M}\sum_{k=0}^{N}\|x_k^i-\hat{x}_k^i\|_2^2+\lambda R(\theta).
    $$


    Optional regularizer:

    $$
    R(\theta)=\frac{1}{\operatorname{mean}\|f_\theta(x)\|_2+\varepsilon}.
    $$


    BPTT computes gradients through all time steps:

    $$
    s_k=s_{k+1}\frac{\partial\Phi_\theta}{\partial x_k}+\frac{\partial\ell_k}{\partial x_k},
    \qquad
    \nabla_\theta\mathcal{L}=\sum_k s_{k+1}\frac{\partial\Phi_\theta}{\partial\theta}.
    $$


    Notation:

    - $t$: continuous time; $t_0$ start; $T$ final time.
    - $t_k$: sampled time points.
    - $x(t)$: state; $x_k$ sampled state.
    - $x_0$: initial state.
    - $f_\theta$: neural vector field.
    - $M$: number of trajectories.
    - $N$: number of time steps.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.1 Initial-condition sampler

    We sample one initial state per trajectory:

    $$
    u_0 \sim \mathcal{U}[-0.5, 0.5], \qquad
    v_0 \sim \mathcal{U}[-0.5, 0.5], \qquad
    x_0=[u_0, v_0]^\top.
    $$


    The output tensor has shape $(M,2)$.
    """)
    return


@app.cell
def _(jax, jnp):
    def sample_ic(key, n_traj):
        k1, k2 = jax.random.split(key)
        u0 = 0.5 * (2.0 * jax.random.uniform(k1, (n_traj,)) - 1.0)
        v0 = 0.5 * (2.0 * jax.random.uniform(k2, (n_traj,)) - 1.0)
        return jnp.stack([u0, v0], axis=-1)

    return (sample_ic,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.2 Reference vector field

    Physical dynamics used for data generation:

    $$
    f(x)=\begin{bmatrix}v\\-\beta v-(g/\ell)\sin(u)\end{bmatrix}.
    $$
    """)
    return


@app.cell
def _(jnp):
    def pendulum_rhs(x, beta, ell, g):
        u = x[..., 0]
        v = x[..., 1]
        du = v
        dv = -beta * v - (g / ell) * jnp.sin(u)
        return jnp.stack([du, dv], axis=-1)

    return (pendulum_rhs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.3 One RK4 solver step

    For step size $h_k$, classical RK4 uses:

    $$
    \begin{aligned}
    k_1 &= f(x_k), \\
    k_2 &= f\left(x_k+\frac{h_k}{2}k_1\right), \\
    k_3 &= f\left(x_k+\frac{h_k}{2}k_2\right), \\
    k_4 &= f\left(x_k+h_k k_3\right), \\
    x_{k+1} &= x_k+\frac{h_k}{6}(k_1+2k_2+2k_3+k_4).
    \end{aligned}
    $$
    """)
    return


@app.function
def rk4_step(f, x_k, h_k):
    k1 = f(x_k)
    k2 = f(x_k + 0.5 * h_k * k1)
    k3 = f(x_k + 0.5 * h_k * k2)
    k4 = f(x_k + h_k * k3)
    return x_k + (h_k / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.4 Trajectory integrator

    Repeated RK4 updates produce

    $$
    X=[x_0, x_1, \ldots, x_N].
    $$


    We use `jax.lax.scan` for efficient loop execution.
    """)
    return


@app.cell
def _(jax, jnp):
    def rk4_integrate(f, x0, t_grid):
        dts = t_grid[1:] - t_grid[:-1]

        def scan_step(xk, dt):
            x_next = rk4_step(f, xk, dt)
            return x_next, x_next

        _, xs = jax.lax.scan(scan_step, x0, dts)
        return jnp.concatenate([x0[None, ...], xs], axis=0)

    return (rk4_integrate,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.5 He-initialized linear layer

    He initialization for ReLU layers:

    $$
    W \sim \mathcal{N}\left(0,\frac{2}{\text{fan-in}}\right), \qquad b=0.
    $$
    """)
    return


@app.cell
def _(jax, jnp):
    def he_init_layer(key, in_dim, out_dim):
        std = jnp.sqrt(2.0 / float(in_dim))
        w = std * jax.random.normal(key, (in_dim, out_dim))
        b = jnp.zeros((out_dim,), dtype=w.dtype)
        return {"w": w, "b": b}

    return (he_init_layer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.6 He-initialized MLP constructor

    Builds the neural vector field architecture $2\to h\to h\to2$.
    """)
    return


@app.cell
def _(he_init_layer, jax):
    def init_mlp(key, hidden_dim):
        k1, k2, k3 = jax.random.split(key, 3)
        return [
            he_init_layer(k1, 2, hidden_dim),
            he_init_layer(k2, hidden_dim, hidden_dim),
            he_init_layer(k3, hidden_dim, 2),
        ]

    return (init_mlp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.7 Vanilla-initialized linear layer

    Xavier-like uniform initialization:

    $$
    W \sim \mathcal{U}\left[-\sqrt{\frac{6}{\text{fan-in}+\text{fan-out}}},\sqrt{\frac{6}{\text{fan-in}+\text{fan-out}}}\right],\quad b=0.
    $$
    """)
    return


@app.cell
def _(jax, jnp):
    def vanilla_init_layer(key, in_dim, out_dim):
        limit = jnp.sqrt(6.0 / (in_dim + out_dim))
        w = jax.random.uniform(key, (in_dim, out_dim), minval=-limit, maxval=limit)
        b = jnp.zeros((out_dim,), dtype=w.dtype)
        return {"w": w, "b": b}

    return (vanilla_init_layer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.8 Vanilla MLP constructor

    Same architecture as the He version, with vanilla initialization.
    """)
    return


@app.cell
def _(jax, vanilla_init_layer):
    def init_mlp_vanilla(key, hidden_dim):
        k1, k2, k3 = jax.random.split(key, 3)
        return [
            vanilla_init_layer(k1, 2, hidden_dim),
            vanilla_init_layer(k2, hidden_dim, hidden_dim),
            vanilla_init_layer(k3, hidden_dim, 2),
        ]

    return (init_mlp_vanilla,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.9 MLP forward evaluation

    Neural vector field:

    $$
    f_\theta(x)=W_3\,\rho\!\left(W_2\,\rho(W_1x+b_1)+b_2\right)+b_3.
    $$
    """)
    return


@app.cell
def _(jax):
    def mlp_apply(params, x):
        x_shape = x.shape
        h = x.reshape((-1, x_shape[-1]))
        h = jax.nn.relu(h @ params[0]["w"] + params[0]["b"])
        h = jax.nn.relu(h @ params[1]["w"] + params[1]["b"])
        y = h @ params[2]["w"] + params[2]["b"]
        return y.reshape(x_shape)

    return (mlp_apply,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Optimizer and Training Utilities

    We use Adam with optional weight decay, gradient clipping, and plateau-based
    learning-rate reduction.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.1 Zero-valued parameter tree

    Creates zeros with the same structure as parameters.
    """)
    return


@app.cell
def _(jax, jnp):
    def tree_zeros_like(tree):
        return jax.tree_util.tree_map(jnp.zeros_like, tree)

    return (tree_zeros_like,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.2 Tree L2 norm

    Global gradient norm:

    $$
    \|g\|_2=\sqrt{\sum_{\text{leaves}}\sum g^2+\varepsilon}.
    $$
    """)
    return


@app.cell
def _(jax, jnp):
    def tree_l2_norm(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        return jnp.sqrt(sum(jnp.sum(x * x) for x in leaves) + 1e-16)

    return (tree_l2_norm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.3 Adam state initialization

    Initializes first moment $m$, second moment $v$, and step counter $t$.
    """)
    return


@app.cell
def _(tree_zeros_like):
    def adam_init(params):
        return {"m": tree_zeros_like(params), "v": tree_zeros_like(params), "t": 0}

    return (adam_init,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.4 Adam update step

    $$
    m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,
    \qquad
    v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2,
    $$


    $$
    \theta_t=\theta_{t-1}-\alpha\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}.
    $$


    Optional weight decay and gradient clipping are applied before moment updates.
    """)
    return


@app.cell
def _(jax, jnp, tree_l2_norm):
    def adam_step(
        params,
        grads,
        state,
        lr,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
        clip_norm=None,
    ):
        if weight_decay != 0.0:
            grads = jax.tree_util.tree_map(lambda g, p: g + weight_decay * p, grads, params)

        if clip_norm is not None:
            grad_norm = tree_l2_norm(grads)
            scale = jnp.minimum(1.0, clip_norm / (grad_norm + 1e-6))
            grads = jax.tree_util.tree_map(lambda g: g * scale, grads)

        t = state["t"] + 1
        m = jax.tree_util.tree_map(lambda m_prev, g: beta1 * m_prev + (1.0 - beta1) * g, state["m"], grads)
        v = jax.tree_util.tree_map(
            lambda v_prev, g: beta2 * v_prev + (1.0 - beta2) * (g * g),
            state["v"],
            grads,
        )
        m_hat = jax.tree_util.tree_map(lambda m_val: m_val / (1.0 - beta1**t), m)
        v_hat = jax.tree_util.tree_map(lambda v_val: v_val / (1.0 - beta2**t), v)
        new_params = jax.tree_util.tree_map(
            lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps), params, m_hat, v_hat
        )
        return new_params, {"m": m, "v": v, "t": t}

    return (adam_step,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.5 ReduceLROnPlateau scheduler

    If the monitored metric does not improve for `patience` epochs,
    learning rate is multiplied by `factor` (with a minimum floor).
    """)
    return


@app.class_definition
class ReduceLROnPlateau:
    def __init__(self, lr, factor=0.5, patience=100, min_lr=1e-6):
        self.lr = lr
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best = float("inf")
        self.bad_count = 0
        self.min_delta = 1e-12

    def step(self, metric):
        if metric < self.best - self.min_delta:
            self.best = metric
            self.bad_count = 0
        else:
            self.bad_count += 1
            if self.bad_count >= self.patience:
                self.lr = max(self.min_lr, self.lr * self.factor)
                self.bad_count = 0
        return self.lr


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Data, Loss, and BPTT Training

    We now compose simulation, loss computation, BPTT gradients, and parameter updates.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.1 Observation dataset builder

    The function samples initial states, simulates clean trajectories, and adds observation noise.
    """)
    return


@app.cell
def _(jax, jnp, pendulum_rhs, rk4_integrate, sample_ic):
    def make_observation_dataset(
        key_ic,
        key_noise,
        g_true,
        beta_true,
        ell_true,
        t_final,
        n_steps,
        n_traj,
        noise_std,
    ):
        t_grid = jnp.linspace(0.0, t_final, n_steps)
        x0_batch = sample_ic(key_ic, n_traj)
        x_true = rk4_integrate(lambda x: pendulum_rhs(x, beta_true, ell_true, g_true), x0_batch, t_grid)
        x_obs = x_true + noise_std * jax.random.normal(key_noise, x_true.shape)
        return t_grid, x0_batch, x_true, x_obs

    return (make_observation_dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.2 Optional normalization

    $$
    \tilde{x}=\frac{x-\mu_{\hat{x}}}{\sigma_{\hat{x}}+\varepsilon}.
    $$


    This often improves optimization conditioning.
    """)
    return


@app.cell
def _(jnp):
    def compute_normalization_stats(x0_batch, x_obs, use_data_normalization):
        if use_data_normalization:
            x_obs_mean = jnp.mean(x_obs, axis=(0, 1), keepdims=True)
            x_obs_std = jnp.std(x_obs, axis=(0, 1), keepdims=True) + 1e-6
            x0_train = (x0_batch - x_obs_mean[0]) / x_obs_std[0]
        else:
            x_obs_mean = jnp.zeros((1, 1, 2), dtype=x_obs.dtype)
            x_obs_std = jnp.ones((1, 1, 2), dtype=x_obs.dtype)
            x0_train = x0_batch
        return x_obs_mean, x_obs_std, x0_train

    return (compute_normalization_stats,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.3 Optional derivative regularization

    $$
    R(\theta)=\frac{1}{\operatorname{mean}\|f_\theta(x)\|_2+\varepsilon},
    \qquad
    \mathcal{L}=\mathcal{L}_{\text{data}}+\lambda R(\theta).
    $$
    """)
    return


@app.cell
def _(jnp):
    def derivative_regularization_term(vector_field_apply, params, x_state, reg_strength, use_regularization):
        if not use_regularization:
            zero = jnp.array(0.0, dtype=x_state.dtype)
            nan_value = jnp.array(jnp.nan, dtype=x_state.dtype)
            return zero, nan_value

        mean_norm = jnp.mean(jnp.linalg.norm(vector_field_apply(params, x_state), axis=-1))
        reg_term = reg_strength * (1.0 / (mean_norm + 1e-6))
        return reg_term, mean_norm

    return (derivative_regularization_term,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.4 Full BPTT training routine

    Each epoch performs forward rollout, objective evaluation, gradient computation,
    and optimizer update:

    $$
    \theta \leftarrow \theta - \alpha\nabla_\theta\mathcal{L}.
    $$
    """)
    return


@app.cell
def _(
    adam_init,
    adam_step,
    compute_normalization_stats,
    derivative_regularization_term,
    init_mlp,
    init_mlp_vanilla,
    jax,
    jnp,
    make_observation_dataset,
    mlp_apply,
    mo,
    np,
    rk4_integrate,
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
        use_jit,
        print_every,
    ):
        key = jax.random.PRNGKey(seed)
        key_ic, key_noise, key_model = jax.random.split(key, 3)

        t_grid, x0_batch, x_true, x_obs = make_observation_dataset(
            key_ic=key_ic,
            key_noise=key_noise,
            g_true=g_true,
            beta_true=beta_true,
            ell_true=ell_true,
            t_final=t_final,
            n_steps=n_steps,
            n_traj=n_traj,
            noise_std=noise_std,
        )

        x_obs_mean, x_obs_std, x0_train = compute_normalization_stats(
            x0_batch=x0_batch,
            x_obs=x_obs,
            use_data_normalization=use_data_normalization,
        )

        if use_he_init:
            params = init_mlp(key_model, hidden_dim)
        else:
            params = init_mlp_vanilla(key_model, hidden_dim)

        state = adam_init(params)
        scheduler = ReduceLROnPlateau(lr=lr, factor=scheduler_factor, patience=scheduler_patience)
        wd = weight_decay if use_weight_decay else 0.0
        clip_value = clip_norm if use_grad_clip else None

        def simulate(curr_params, x0):
            return rk4_integrate(lambda x: mlp_apply(curr_params, x), x0, t_grid)

        def objective(curr_params):
            x_sim = simulate(curr_params, x0_train)
            x_sim_rescaled = x_sim * x_obs_std + x_obs_mean
            mse_loss = jnp.mean((x_sim_rescaled - x_obs) ** 2)
            reg_term, deriv_norm = derivative_regularization_term(
                vector_field_apply=mlp_apply,
                params=curr_params,
                x_state=x_sim,
                reg_strength=reg_strength,
                use_regularization=use_regularization,
            )
            loss = mse_loss + reg_term
            return loss, (mse_loss, deriv_norm, reg_term)

        def train_step(curr_params, curr_state, curr_lr):
            (loss, (mse_loss, deriv_norm, reg_term)), grads = jax.value_and_grad(objective, has_aux=True)(curr_params)
            next_params, next_state = adam_step(
                curr_params,
                grads,
                curr_state,
                lr=curr_lr,
                weight_decay=wd,
                clip_norm=clip_value,
            )
            return next_params, next_state, loss, mse_loss, deriv_norm, reg_term

        if use_jit:
            train_step = jax.jit(train_step)
            simulate_eval = jax.jit(simulate)
        else:
            simulate_eval = simulate

        total_history = []
        mse_history = []
        reg_history = []
        deriv_history = []
        lr_history = []

        for epoch in range(epochs):
            current_lr = scheduler.lr if use_scheduler else lr
            params, state, loss, mse_loss, deriv_norm, reg_term = train_step(params, state, current_lr)

            loss_value = float(loss)
            mse_value = float(mse_loss)
            reg_value = float(reg_term)
            deriv_value = float(deriv_norm)

            if not np.isfinite(loss_value):
                print(f"[{epoch:04d}] non-finite loss; stopping early")
                break

            if use_scheduler:
                current_lr = scheduler.step(mse_value)

            total_history.append(loss_value)
            mse_history.append(mse_value)
            reg_history.append(reg_value)
            deriv_history.append(deriv_value)
            lr_history.append(float(current_lr))

            if epoch % print_every == 0:
                print(
                    f"[{epoch:04d}] total={loss_value:.6f} "
                    f"mse={mse_value:.6f} reg={reg_value:.6f} "
                    f"lr={current_lr:.6f}"
                )

        x_fit = simulate_eval(params, x0_train)
        x_fit = x_fit * x_obs_std + x_obs_mean
        final_clean_mse = float(jnp.mean((x_fit - x_true) ** 2))

        return {
            "t_grid": np.asarray(t_grid),
            "x_true": np.asarray(x_true),
            "x_obs": np.asarray(x_obs),
            "x_fit": np.asarray(x_fit),
            "total_history": np.asarray(total_history),
            "mse_history": np.asarray(mse_history),
            "reg_history": np.asarray(reg_history),
            "deriv_history": np.asarray(deriv_history),
            "lr_history": np.asarray(lr_history),
            "final_clean_mse": final_clean_mse,
        }

    return (train_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a id="control-panel"></a>
    ## 4. Control Panel

    **Vanilla baseline (default):**

    - `Use regularization term` off.
    - normalization, weight decay, scheduler, clipping, and He init off.
    - `hidden_dim=64`, `lr=5e-2`, `epochs=501`.

    **Tuning sequence:**

    1. Turn on `Use regularization term`, start with `reg_strength = 0.01`.
    2. If trajectories become too flat, decrease regularization.
    3. If optimization oscillates, lower `lr` and enable clipping.
    4. Then enable normalization, He init, weight decay, and scheduler.
    5. Larger `N` improves time resolution; larger `M` improves state coverage.
    """)
    return


@app.cell(hide_code=True)
def _(jax, mo):
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
    use_jit = mo.ui.checkbox(label="Use jax.jit", value=True)

    plot_count_slider = mo.ui.slider(1, 8, value=3, step=1, label="Trajectories to plot")

    train_button = mo.ui.run_button(label="Train JAX BPTT NODE")

    panel = mo.vstack(
        [
            train_button,
            mo.md(f"**JAX backend:** `{jax.default_backend()}`"),
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
            use_jit,
            mo.md("### Visualization"),
            plot_count_slider,
        ]
    )

    panel
    return (
        beta_slider,
        clip_norm_slider,
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
        use_jit,
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
    use_jit,
    use_norm,
    use_regularization,
    use_scheduler,
    use_weight_decay,
    weight_decay_dropdown,
):
    mo.stop(not train_button.value, mo.md("_Click **Train JAX BPTT NODE** to start training._"))

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
        use_jit=bool(use_jit.value),
        print_every=int(print_every_slider.value),
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

    We compare predicted and reference trajectories, plus component-wise errors.
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
    ### 5.2 Optimization history

    Displays total loss, data loss, regularization term, and learning-rate schedule.
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
