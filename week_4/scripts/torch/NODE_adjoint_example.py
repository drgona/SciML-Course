import matplotlib.pyplot as plt
import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G_TRUE = 9.81
BETA_TRUE = 0.25
ELL_TRUE = 0.9
T = 5.0
N = 200
M = 32
NOISE_STD = 0.01
HIDDEN_DIM = 128
LR = 1e-2
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 1000
PRINT_EVERY = 100
PLOT_COUNT = 3


def sample_ic(num_traj: int) -> torch.Tensor:
    u0 = 0.5 * (2 * torch.rand(num_traj, device=device) - 1)
    v0 = 0.5 * (2 * torch.rand(num_traj, device=device) - 1)
    return torch.stack([u0, v0], dim=-1)


def pendulum_rhs(x: torch.Tensor, beta: float, ell: float, g: float = 9.81) -> torch.Tensor:
    u, v = x[..., 0], x[..., 1]
    return torch.stack([v, -beta * v - (g / ell) * torch.sin(u)], dim=-1)


def rk4_step_autonomous(f, x: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def rk4_integrate_autonomous(f, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    x = x0
    xs = [x]
    for k in range(t.shape[0] - 1):
        dt = t[k + 1] - t[k]
        x = rk4_step_autonomous(f, x, dt)
        xs.append(x)
    return torch.stack(xs, dim=0)


class ODEFunc(nn.Module):
    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        ).to(device)
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _augmented_dynamics(func, x: torch.Tensor, a: torch.Tensor, params):
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
    dpdt = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads[1:], params)]
    return f.detach(), dadt.detach(), [g.detach() for g in dpdt]


def _rk4_augmented_step(func, x: torch.Tensor, a: torch.Tensor, dt: torch.Tensor, params):
    k1x, k1a, k1p = _augmented_dynamics(func, x, a, params)
    k2x, k2a, k2p = _augmented_dynamics(func, x + 0.5 * dt * k1x, a + 0.5 * dt * k1a, params)
    k3x, k3a, k3p = _augmented_dynamics(func, x + 0.5 * dt * k2x, a + 0.5 * dt * k2a, params)
    k4x, k4a, k4p = _augmented_dynamics(func, x + dt * k3x, a + dt * k3a, params)

    x_new = x + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
    a_new = a + (dt / 6.0) * (k1a + 2 * k2a + 2 * k3a + k4a)
    gp = []
    for j in range(len(params)):
        gp.append((dt / 6.0) * (k1p[j] + 2 * k2p[j] + 2 * k3p[j] + k4p[j]))
    return x_new, a_new, gp


class ODEAdjointRK4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x0: torch.Tensor, t: torch.Tensor, func, *params):
        with torch.no_grad():
            y = rk4_integrate_autonomous(func, x0, t)
        ctx.func = func
        ctx.save_for_backward(t, y, *params)
        return y

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        t, y, *params = ctx.saved_tensors
        func = ctx.func

        grad_params = [torch.zeros_like(p) for p in params]
        a = grad_y[-1]

        for i in range(t.shape[0] - 1, 0, -1):
            dt = t[i - 1] - t[i]
            _, a, gp = _rk4_augmented_step(func, y[i], a, dt, params)
            for j in range(len(grad_params)):
                grad_params[j] = grad_params[j] + gp[j]
            a = a + grad_y[i - 1]

        return a, None, None, *grad_params


class NeuralODEAdjoint(nn.Module):
    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.func = ODEFunc(hidden_dim=hidden_dim)

    def forward(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        params = tuple(self.func.parameters())
        return ODEAdjointRK4.apply(x0, t, self.func, *params)


def compute_derivative_norm(model_ref: NeuralODEAdjoint, x_sim: torch.Tensor) -> torch.Tensor:
    with torch.enable_grad():
        x_eval = x_sim.clone().requires_grad_(True)
        dx_dt = model_ref.func.net(x_eval)
        return torch.mean(torch.norm(dx_dt, dim=-1))


def plot_trajectories(
    t: torch.Tensor,
    x_ref: torch.Tensor,
    x_pred: torch.Tensor,
    num_trajectories: int = PLOT_COUNT,
    save_name: str = "NODE_adjoint_example_plot.png",
) -> None:
    t_np = t.cpu().numpy()
    x_ref_np = x_ref.cpu().numpy()
    x_pred_np = x_pred.cpu().numpy()
    idx = torch.arange(min(num_trajectories, x_ref.shape[1])).cpu().numpy()

    fig, axes = plt.subplots(3, len(idx), figsize=(5 * len(idx), 11), sharex=True, squeeze=False)
    for i, k in enumerate(idx):
        axes[0, i].plot(t_np, x_ref_np[:, k, 0], "k-", label="True u")
        axes[0, i].plot(t_np, x_pred_np[:, k, 0], "r--", label="Adjoint NODE u")
        axes[0, i].set_title(f"Trajectory {k}")
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].legend()

        axes[1, i].plot(t_np, x_ref_np[:, k, 1], "k-", label="True v")
        axes[1, i].plot(t_np, x_pred_np[:, k, 1], "r--", label="Adjoint NODE v")
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].legend()

        axes[2, i].plot(t_np, x_pred_np[:, k, 0] - x_ref_np[:, k, 0], "m-", label="u error")
        axes[2, i].plot(t_np, x_pred_np[:, k, 1] - x_ref_np[:, k, 1], "g--", label="v error")
        axes[2, i].axhline(0.0, color="k", linewidth=1.0, alpha=0.5)
        axes[2, i].set_xlabel("Time (s)")
        axes[2, i].set_title(f"Trajectory {k}: error")
        axes[2, i].grid(True, alpha=0.3)
        axes[2, i].legend()

    plt.tight_layout()
    fig.savefig(save_name, dpi=180, bbox_inches="tight")
    print(f"saved_plot={save_name}")
    plt.show()


t_grid = torch.linspace(0.0, T, N, device=device)
x0_batch = sample_ic(M)
with torch.no_grad():
    x_true = rk4_integrate_autonomous(lambda x: pendulum_rhs(x, BETA_TRUE, ELL_TRUE, G_TRUE), x0_batch, t_grid)
x_obs = (x_true + NOISE_STD * torch.randn_like(x_true)).detach()
x_obs_mean = x_obs.mean(dim=[0, 1], keepdim=True)
x_obs_std = x_obs.std(dim=[0, 1], keepdim=True) + 1e-6
x0_batch_normalized = (x0_batch - x_obs_mean[0]) / x_obs_std[0]

model = NeuralODEAdjoint(hidden_dim=HIDDEN_DIM).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
try:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=100, verbose=True
    )
except TypeError:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=100)
mse = nn.MSELoss()

for epoch in range(NUM_EPOCHS):
    optimizer.zero_grad()
    x_sim = model(x0_batch_normalized, t_grid)
    x_sim_rescaled = x_sim * x_obs_std + x_obs_mean
    mse_loss = mse(x_sim_rescaled, x_obs)
    deriv_norm = compute_derivative_norm(model, x_sim)
    loss = mse_loss + 0.01 * (1.0 / (deriv_norm + 1e-6))
    if not torch.isfinite(loss):
        print(f"[{epoch:04d}] loss became non-finite; stopping early")
        break
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step(mse_loss.detach())

    if epoch % PRINT_EVERY == 0:
        print(
            f"[{epoch:04d}] mse_loss={mse_loss.item():.6f} "
            f"deriv_norm={deriv_norm.item():.6f} lr={optimizer.param_groups[0]['lr']:.6f}"
        )

with torch.no_grad():
    x_fit = model(x0_batch_normalized, t_grid)
    x_fit = x_fit * x_obs_std + x_obs_mean
    final_mse = mse(x_fit, x_true).item()
print(f"final_clean_mse={final_mse:.6f}")

plot_trajectories(t_grid, x_true, x_fit, num_trajectories=PLOT_COUNT)
