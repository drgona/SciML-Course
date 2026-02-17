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
HIDDEN_DIM = 64
LR = 5e-2
NUM_EPOCHS = 501
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
    num_trajectories: int = PLOT_COUNT,
    save_name: str = "NODE_BPTT_vanilla_plot.png",
) -> None:
    t_np = np.asarray(t_grid)
    x_true_np = np.asarray(x_true)
    x_fit_np = np.asarray(x_fit)
    indices = np.arange(min(num_trajectories, x_true_np.shape[1]))

    fig, axes = plt.subplots(3, len(indices), figsize=(5 * len(indices), 11), sharex=True, squeeze=False)
    for i, idx in enumerate(indices):
        axes[0, i].plot(t_np, x_true_np[:, idx, 0], "b-", label="True u")
        axes[0, i].plot(t_np, x_fit_np[:, idx, 0], "r--", label="NODE u")
        axes[0, i].set_title(f"Trajectory {idx + 1}: u")
        axes[0, i].grid(True)
        axes[0, i].legend()

        axes[1, i].plot(t_np, x_true_np[:, idx, 1], "b-", label="True v")
        axes[1, i].plot(t_np, x_fit_np[:, idx, 1], "r--", label="NODE v")
        axes[1, i].set_title(f"Trajectory {idx + 1}: v")
        axes[1, i].grid(True)
        axes[1, i].legend()

        axes[2, i].plot(t_np, x_fit_np[:, idx, 0] - x_true_np[:, idx, 0], "m-", label="u error")
        axes[2, i].plot(t_np, x_fit_np[:, idx, 1] - x_true_np[:, idx, 1], "g--", label="v error")
        axes[2, i].axhline(0.0, color="k", linewidth=1.0, alpha=0.5)
        axes[2, i].set_title(f"Trajectory {idx + 1}: error")
        axes[2, i].set_xlabel("Time (s)")
        axes[2, i].grid(True)
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

    params = init_mlp(key_model, hidden_dim=HIDDEN_DIM)
    state = adam_init(params)

    def simulate(curr_params, x0):
        return rk4_integrate(lambda x: mlp_apply(curr_params, x), x0, t_grid)

    def loss_fn(curr_params):
        x_sim = simulate(curr_params, x0_batch)
        return jnp.mean((x_sim - x_obs) ** 2)

    def train_step(curr_params, curr_state):
        loss, grads = jax.value_and_grad(loss_fn)(curr_params)
        next_params, next_state = adam_step(curr_params, grads, curr_state, lr=LR)
        return next_params, next_state, loss

    if USE_JIT:
        train_step_jit = jax.jit(train_step)
        simulate_eval = jax.jit(simulate)
    else:
        train_step_jit = train_step
        simulate_eval = simulate

    for epoch in range(NUM_EPOCHS):
        params, state, loss = train_step_jit(params, state)
        if epoch % PRINT_EVERY == 0:
            print(f"[{epoch:04d}] loss={float(loss):.6f}")

    x_fit = simulate_eval(params, x0_batch)
    plot_trajectories(t_grid=t_grid, x_true=x_true, x_fit=x_fit, num_trajectories=PLOT_COUNT)


if __name__ == "__main__":
    main()
