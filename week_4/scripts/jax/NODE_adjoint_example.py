import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


SEED = 0
USE_JIT = True
G_TRUE = 9.81
BETA_TRUE = 0.25
ELL_TRUE = 0.9
T_FINAL = 5.0
N_STEPS = 200
N_TRAJ = 32
NOISE_STD = 0.01
HIDDEN_DIM = 128
LR = 1e-2
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 1000
PRINT_EVERY = 100
PLOT_COUNT = 3


def pendulum_rhs(x: jnp.ndarray, beta: float, ell: float, g: float = 9.81) -> jnp.ndarray:
    u = x[..., 0]
    v = x[..., 1]
    return jnp.stack([v, -beta * v - (g / ell) * jnp.sin(u)], axis=-1)


def rk4_step(f, x: jnp.ndarray, dt: jnp.ndarray) -> jnp.ndarray:
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk4_integrate(f, x0: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    dts = t[1:] - t[:-1]

    def scan_step(xk, dt):
        x_next = rk4_step(f, xk, dt)
        return x_next, x_next

    _, xs = jax.lax.scan(scan_step, x0, dts)
    return jnp.concatenate([x0[None, ...], xs], axis=0)


def sample_ic(key: jax.Array, n_traj: int) -> jnp.ndarray:
    k1, k2 = jax.random.split(key)
    u0 = 0.5 * (2.0 * jax.random.uniform(k1, (n_traj,)) - 1.0)
    v0 = 0.5 * (2.0 * jax.random.uniform(k2, (n_traj,)) - 1.0)
    return jnp.stack([u0, v0], axis=-1)


def he_init_layer(key: jax.Array, in_dim: int, out_dim: int) -> dict[str, jnp.ndarray]:
    std = jnp.sqrt(2.0 / float(in_dim))
    w = std * jax.random.normal(key, (in_dim, out_dim))
    b = jnp.zeros((out_dim,), dtype=w.dtype)
    return {"w": w, "b": b}


def init_mlp(key: jax.Array, hidden_dim: int) -> list[dict[str, jnp.ndarray]]:
    k1, k2, k3 = jax.random.split(key, 3)
    return [
        he_init_layer(k1, 2, hidden_dim),
        he_init_layer(k2, hidden_dim, hidden_dim),
        he_init_layer(k3, hidden_dim, 2),
    ]


def mlp_apply(params: list[dict[str, jnp.ndarray]], x: jnp.ndarray) -> jnp.ndarray:
    x_shape = x.shape
    h = x.reshape((-1, x_shape[-1]))
    h = jax.nn.relu(h @ params[0]["w"] + params[0]["b"])
    h = jax.nn.relu(h @ params[1]["w"] + params[1]["b"])
    y = h @ params[2]["w"] + params[2]["b"]
    return y.reshape(x_shape)


def tree_add(a, b):
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def tree_zeros_like(tree):
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def tree_l2_norm(tree) -> jnp.ndarray:
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(x * x) for x in leaves) + 1e-16)


def adam_init(params):
    return {"m": tree_zeros_like(params), "v": tree_zeros_like(params), "t": 0}


def adam_step(
    params,
    grads,
    state,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    clip_norm: float | None = None,
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


class ReduceLROnPlateau:
    def __init__(self, lr: float, factor: float = 0.5, patience: int = 100, min_lr: float = 1e-6):
        self.lr = lr
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best = float("inf")
        self.bad_count = 0
        self.min_delta = 1e-12

    def step(self, metric: float) -> float:
        if metric < self.best - self.min_delta:
            self.best = metric
            self.bad_count = 0
        else:
            self.bad_count += 1
            if self.bad_count >= self.patience:
                self.lr = max(self.min_lr, self.lr * self.factor)
                self.bad_count = 0
        return self.lr


def integrate_forward(params, x0: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    return rk4_integrate(lambda x: mlp_apply(params, x), x0, t)


@jax.custom_vjp
def integrate_adjoint(params, x0: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    return integrate_forward(params, x0, t)


def integrate_adjoint_fwd(params, x0: jnp.ndarray, t: jnp.ndarray):
    y = integrate_forward(params, x0, t)
    return y, (params, t, y)


def integrate_adjoint_bwd(res, g_y: jnp.ndarray):
    params, t, y = res
    x_curr_rev = y[1:][::-1]
    g_prev_rev = g_y[:-1][::-1]
    dt_rev = (t[:-1] - t[1:])[::-1]

    def augmented_dynamics(p, x: jnp.ndarray, a: jnp.ndarray):
        def rhs_fn(pp, xx):
            return mlp_apply(pp, xx)

        f = rhs_fn(p, x)
        _, pullback = jax.vjp(rhs_fn, p, x)
        dadt_p, dadt_x = pullback(-a)
        return f, dadt_x, dadt_p

    def rk4_augmented_step(p, x: jnp.ndarray, a: jnp.ndarray, dt: jnp.ndarray):
        k1x, k1a, k1p = augmented_dynamics(p, x, a)
        k2x, k2a, k2p = augmented_dynamics(p, x + 0.5 * dt * k1x, a + 0.5 * dt * k1a)
        k3x, k3a, k3p = augmented_dynamics(p, x + 0.5 * dt * k2x, a + 0.5 * dt * k2a)
        k4x, k4a, k4p = augmented_dynamics(p, x + dt * k3x, a + dt * k3a)

        x_new = x + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        a_new = a + (dt / 6.0) * (k1a + 2.0 * k2a + 2.0 * k3a + k4a)
        gp = jax.tree_util.tree_map(
            lambda p1, p2, p3, p4: (dt / 6.0) * (p1 + 2.0 * p2 + 2.0 * p3 + p4),
            k1p,
            k2p,
            k3p,
            k4p,
        )
        return x_new, a_new, gp

    def scan_step(carry, inp):
        a, grad_params = carry
        x_curr, g_prev, dt = inp
        _, a_int, gp = rk4_augmented_step(params, x_curr, a, dt)
        next_a = a_int + g_prev
        next_grad_params = tree_add(grad_params, gp)
        return (next_a, next_grad_params), None

    init_carry = (g_y[-1], tree_zeros_like(params))
    (grad_x0, grad_params), _ = jax.lax.scan(scan_step, init_carry, (x_curr_rev, g_prev_rev, dt_rev))
    grad_t = jnp.zeros_like(t)
    return grad_params, grad_x0, grad_t


integrate_adjoint.defvjp(integrate_adjoint_fwd, integrate_adjoint_bwd)


def plot_trajectories(
    t_grid: jnp.ndarray,
    x_true: jnp.ndarray,
    x_fit: jnp.ndarray,
    num_trajectories: int = PLOT_COUNT,
    save_name: str = "NODE_adjoint_example_plot.png",
) -> None:
    t_np = np.asarray(t_grid)
    x_true_np = np.asarray(x_true)
    x_fit_np = np.asarray(x_fit)
    indices = np.arange(min(num_trajectories, x_true_np.shape[1]))

    fig, axes = plt.subplots(3, len(indices), figsize=(5 * len(indices), 11), sharex=True, squeeze=False)
    for i, idx in enumerate(indices):
        axes[0, i].plot(t_np, x_true_np[:, idx, 0], "k-", label="True u")
        axes[0, i].plot(t_np, x_fit_np[:, idx, 0], "r--", label="Adjoint NODE u")
        axes[0, i].set_title(f"Trajectory {idx}")
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].legend()

        axes[1, i].plot(t_np, x_true_np[:, idx, 1], "k-", label="True v")
        axes[1, i].plot(t_np, x_fit_np[:, idx, 1], "r--", label="Adjoint NODE v")
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].legend()

        axes[2, i].plot(t_np, x_fit_np[:, idx, 0] - x_true_np[:, idx, 0], "m-", label="u error")
        axes[2, i].plot(t_np, x_fit_np[:, idx, 1] - x_true_np[:, idx, 1], "g--", label="v error")
        axes[2, i].axhline(0.0, color="k", linewidth=1.0, alpha=0.5)
        axes[2, i].set_xlabel("Time (s)")
        axes[2, i].set_title(f"Trajectory {idx}: error")
        axes[2, i].grid(True, alpha=0.3)
        axes[2, i].legend()

    plt.tight_layout()
    fig.savefig(save_name, dpi=180, bbox_inches="tight")
    print(f"saved_plot={save_name}")
    plt.show()


def main() -> None:
    t_grid = jnp.linspace(0.0, T_FINAL, N_STEPS)
    print(f"backend={jax.default_backend()} devices={jax.devices()} jit={USE_JIT}")

    key = jax.random.PRNGKey(SEED)
    key_ic, key_noise, key_model = jax.random.split(key, 3)

    x0_batch = sample_ic(key_ic, N_TRAJ)
    x_true = rk4_integrate(lambda x: pendulum_rhs(x, BETA_TRUE, ELL_TRUE, G_TRUE), x0_batch, t_grid)
    x_obs = x_true + NOISE_STD * jax.random.normal(key_noise, x_true.shape)

    x_obs_mean = jnp.mean(x_obs, axis=(0, 1), keepdims=True)
    x_obs_std = jnp.std(x_obs, axis=(0, 1), keepdims=True) + 1e-6
    x0_batch_normalized = (x0_batch - x_obs_mean[0]) / x_obs_std[0]

    params = init_mlp(key_model, hidden_dim=HIDDEN_DIM)
    state = adam_init(params)
    scheduler = ReduceLROnPlateau(lr=LR, factor=0.5, patience=100)

    def simulate(curr_params, x0):
        return integrate_adjoint(curr_params, x0, t_grid)

    def objective(curr_params):
        x_sim = simulate(curr_params, x0_batch_normalized)
        x_sim_rescaled = x_sim * x_obs_std + x_obs_mean
        mse_loss = jnp.mean((x_sim_rescaled - x_obs) ** 2)
        deriv_norm = jnp.mean(jnp.linalg.norm(mlp_apply(curr_params, x_sim), axis=-1))
        loss = mse_loss + 0.01 * (1.0 / (deriv_norm + 1e-6))
        return loss, (mse_loss, deriv_norm)

    def train_step(curr_params, curr_state, curr_lr):
        (loss, (mse_loss, deriv_norm)), grads = jax.value_and_grad(objective, has_aux=True)(curr_params)
        next_params, next_state = adam_step(
            curr_params,
            grads,
            curr_state,
            lr=curr_lr,
            weight_decay=WEIGHT_DECAY,
            clip_norm=1.0,
        )
        return next_params, next_state, loss, mse_loss, deriv_norm

    if USE_JIT:
        train_step_jit = jax.jit(train_step)
        simulate_eval = jax.jit(simulate)
    else:
        train_step_jit = train_step
        simulate_eval = simulate

    for epoch in range(NUM_EPOCHS):
        curr_lr = scheduler.lr
        params, state, loss, mse_loss, deriv_norm = train_step_jit(params, state, curr_lr)
        if not jnp.isfinite(loss):
            print(f"[{epoch:04d}] loss became non-finite; stopping early")
            break
        curr_lr = scheduler.step(float(mse_loss))
        if epoch % PRINT_EVERY == 0:
            print(
                f"[{epoch:04d}] mse_loss={float(mse_loss):.6f} "
                f"deriv_norm={float(deriv_norm):.6f} lr={curr_lr:.6f}"
            )

    x_fit = simulate_eval(params, x0_batch_normalized)
    x_fit = x_fit * x_obs_std + x_obs_mean
    final_mse = jnp.mean((x_fit - x_true) ** 2)
    print(f"final_clean_mse={float(final_mse):.6f}")

    plot_trajectories(t_grid=t_grid, x_true=x_true, x_fit=x_fit, num_trajectories=PLOT_COUNT)


if __name__ == "__main__":
    main()
