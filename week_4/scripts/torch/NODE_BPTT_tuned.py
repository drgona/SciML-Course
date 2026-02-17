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


def rk4_integrate(f, x0: torch.Tensor, t: torch.Tensor, *f_args) -> torch.Tensor:
    x = x0
    xs = [x]
    for k in range(t.shape[0] - 1):
        dt = t[k + 1] - t[k]
        k1 = f(x, *f_args)
        k2 = f(x + 0.5 * dt * k1, *f_args)
        k3 = f(x + 0.5 * dt * k2, *f_args)
        k4 = f(x + dt * k3, *f_args)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        xs.append(x)
    return torch.stack(xs, dim=0)


class NeuralODE(nn.Module):
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

    def forward(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return rk4_integrate(lambda x: self.net(x), x0, t)


def compute_derivative_norm(model: NeuralODE, x_sim: torch.Tensor) -> torch.Tensor:
    with torch.enable_grad():
        x_eval = x_sim.clone().requires_grad_(True)
        dx_dt = model.net(x_eval)
        return torch.mean(torch.norm(dx_dt, dim=-1))


def plot_trajectories(
    t_grid: torch.Tensor,
    x_true: torch.Tensor,
    x_fit: torch.Tensor,
    num_trajectories: int = PLOT_COUNT,
    save_name: str = "NODE_BPTT_tuned_plot.png",
) -> None:
    t_np = t_grid.cpu().numpy()
    x_true_np = x_true.cpu().numpy()
    x_fit_np = x_fit.cpu().numpy()
    indices = torch.arange(min(num_trajectories, x_true.shape[1])).cpu().numpy()

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


t_grid = torch.linspace(0.0, T, N, device=device)
x0_batch = sample_ic(M)
with torch.no_grad():
    x_true = rk4_integrate(lambda x, b, l: pendulum_rhs(x, b, l, G_TRUE), x0_batch, t_grid, BETA_TRUE, ELL_TRUE)
x_obs = (x_true + NOISE_STD * torch.randn_like(x_true)).detach()

x_obs_mean = x_obs.mean(dim=[0, 1], keepdim=True)
x_obs_std = x_obs.std(dim=[0, 1], keepdim=True) + 1e-6
x0_batch_normalized = (x0_batch - x_obs_mean[0]) / x_obs_std[0]

model = NeuralODE(hidden_dim=HIDDEN_DIM).to(device)
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

plot_trajectories(t_grid, x_true, x_fit, num_trajectories=PLOT_COUNT)
