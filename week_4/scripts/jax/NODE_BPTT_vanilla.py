import os
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


SEED = int(os.getenv("SEED", "0"))
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(SCRIPT_DIR)))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SHOW_PLOT = os.getenv("SHOW_PLOT", "1") == "1"
PLOT_COUNT = int(os.getenv("PLOT_COUNT", "3"))
USE_JIT = os.getenv("USE_JIT", "1") == "1"


def pendulum_rhs(x: jnp.ndarray, beta: float, ell: float, g: float = 9.81) -> jnp.ndarray:
    u = x[..., 0]
    v = x[..., 1]
    du = v
    dv = -beta * v - (g / ell) * jnp.sin(u)
    return jnp.stack([du, dv], axis=-1)


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


def plot_trajectories(
    t_grid: jnp.ndarray,
    x_true: jnp.ndarray,
    x_fit: jnp.ndarray,
    num_trajectories: int,
    save_dir: Path,
    save_name: str,
    show_plot: bool,
):
    t_np = np.asarray(t_grid)
    x_true_np = np.asarray(x_true)
    x_fit_np = np.asarray(x_fit)
    indices = np.arange(min(num_trajectories, x_true_np.shape[1]))

    fig, axes = plt.subplots(
        3, len(indices), figsize=(5 * len(indices), 11), sharex=True, squeeze=False
    )
    for i, idx in enumerate(indices):
        axes[0, i].plot(t_np, x_true_np[:, idx, 0], "b-", label="True Angle (u)")
        axes[0, i].plot(t_np, x_fit_np[:, idx, 0], "r--", label="Fitted Angle (u)")
        axes[0, i].set_xlabel("Time (s)")
        axes[0, i].set_ylabel("Angle (rad)")
        axes[0, i].set_title(f"Trajectory {idx + 1}: Angle")
        axes[0, i].legend()
        axes[0, i].grid(True)

        axes[1, i].plot(t_np, x_true_np[:, idx, 1], "b-", label="True Angular Velocity (v)")
        axes[1, i].plot(t_np, x_fit_np[:, idx, 1], "r--", label="Fitted Angular Velocity (v)")
        axes[1, i].set_xlabel("Time (s)")
        axes[1, i].set_ylabel("Angular Velocity (rad/s)")
        axes[1, i].set_title(f"Trajectory {idx + 1}: Angular Velocity")
        axes[1, i].legend()
        axes[1, i].grid(True)

        err_u = x_fit_np[:, idx, 0] - x_true_np[:, idx, 0]
        err_v = x_fit_np[:, idx, 1] - x_true_np[:, idx, 1]
        axes[2, i].plot(t_np, err_u, "m-", label="Error u_hat - u")
        axes[2, i].plot(t_np, err_v, "g--", label="Error v_hat - v")
        axes[2, i].axhline(0.0, color="k", linewidth=1.0, alpha=0.5)
        axes[2, i].set_xlabel("Time (s)")
        axes[2, i].set_ylabel("State Error")
        axes[2, i].set_title(f"Trajectory {idx + 1}: Error")
        axes[2, i].legend()
        axes[2, i].grid(True)

    plt.tight_layout()
    save_path = save_dir / save_name
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    print(f"saved_plot={save_path}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def main():
    g_true = 9.81
    beta_true = 0.25
    ell_true = 0.9

    t_final = float(os.getenv("T_FINAL", "5.0"))
    n_steps = int(os.getenv("N_STEPS", "200"))
    n_traj = int(os.getenv("N_TRAJ", "32"))
    t_grid = jnp.linspace(0.0, t_final, n_steps)
    print(f"backend={jax.default_backend()} devices={jax.devices()} jit={USE_JIT}")

    key = jax.random.PRNGKey(SEED)
    key_ic, key_noise, key_model = jax.random.split(key, 3)

    x0_batch = sample_ic(key_ic, n_traj)
    x_true = rk4_integrate(lambda x: pendulum_rhs(x, beta_true, ell_true, g_true), x0_batch, t_grid)
    x_obs = x_true + 0.01 * jax.random.normal(key_noise, x_true.shape)

    params = init_mlp(key_model, hidden_dim=64)
    state = adam_init(params)

    def simulate(curr_params, x0):
        return rk4_integrate(lambda x: mlp_apply(curr_params, x), x0, t_grid)

    def loss_fn(curr_params):
        x_sim = simulate(curr_params, x0_batch)
        return jnp.mean((x_sim - x_obs) ** 2)

    def train_step(curr_params, curr_state):
        loss, grads = jax.value_and_grad(loss_fn)(curr_params)
        next_params, next_state = adam_step(curr_params, grads, curr_state, lr=5e-2)
        return next_params, next_state, loss

    if USE_JIT:
        train_step = jax.jit(train_step)
        simulate_eval = jax.jit(simulate)
    else:
        simulate_eval = simulate

    num_epochs = int(os.getenv("NUM_EPOCHS", "501"))
    print_every = int(os.getenv("PRINT_EVERY", "100"))

    for epoch in range(num_epochs):
        params, state, loss = train_step(params, state)
        if epoch % print_every == 0:
            print(f"[{epoch:04d}] loss={float(loss):.6f}")

    x_fit = simulate_eval(params, x0_batch)
    plot_trajectories(
        t_grid=t_grid,
        x_true=x_true,
        x_fit=x_fit,
        num_trajectories=PLOT_COUNT,
        save_dir=OUTPUT_DIR,
        save_name="NODE_BPTT_vanilla_plot.png",
        show_plot=SHOW_PLOT,
    )


if __name__ == "__main__":
    main()
