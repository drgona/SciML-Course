# Vanilla PyTorch Neural ODE with a black-box neural network for a damped pendulum
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

# -----------------------------
# 1) Problem setup and data gen
# -----------------------------
# State: x = [u, v] with u = angle (rad), v = angular velocity (rad/s)
# Dynamics: du/dt = v
#           dv/dt = -beta * v - (g/ell) * sin(u) (but we'll learn this with a NN)

g_true     = 9.81
beta_true  = 0.25
ell_true   = 0.9

# Time grid (shared across trajectories)
T      = 5.0
N      = 200
t_grid = torch.linspace(0.0, T, N, device=device)  # shape [N]
h      = t_grid[1] - t_grid[0]

# Synthetic dataset: M trajectories with different initial conditions
M = 32
def sample_ic(M):
    # Small random angles/velocities
    u0 = 0.5 * (2*torch.rand(M, device=device)-1)   # ~ [-0.5, 0.5]
    v0 = 0.5 * (2*torch.rand(M, device=device)-1)
    return torch.stack([u0, v0], dim=-1)  # [M, 2]

x0_batch = sample_ic(M)  # [M,2]

def pendulum_rhs(x, beta, ell, g=9.81):
    # x: [..., 2] -> returns [..., 2]
    u, v = x[..., 0], x[..., 1]
    du = v
    dv = -beta * v - (g / ell) * torch.sin(u)
    return torch.stack([du, dv], dim=-1)

def rk4_integrate(f, x0, t, *f_args):
    """
    Runge-Kutta 4 for a batch of trajectories.
    x0: [M,2], t: [N], returns x: [N,M,2]
    Assumes shared time grid for the whole batch.
    """
    M = x0.shape[0]
    N = t.shape[0]
    x = torch.zeros(N, M, 2, device=x0.device, dtype=x0.dtype)  # Initialize output tensor
    x = x.clone()  # Ensure a fresh tensor
    x = torch.index_copy(x, 0, torch.tensor(0, device=x0.device), x0.unsqueeze(0))  # Set initial condition

    for k in range(N-1):
        h = t[k+1] - t[k]
        xk = x[k]
        k1 = f(xk, *f_args)
        k2 = f(xk + 0.5*h*k1, *f_args)
        k3 = f(xk + 0.5*h*k2, *f_args)
        k4 = f(xk + h*k3, *f_args)
        # Create new tensor for the next step instead of inplace update
        x_next = xk + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        x = torch.index_copy(x, 0, torch.tensor(k+1, device=x0.device), x_next.unsqueeze(0))
    return x  # [N, M, 2]

# Generate noise-free ground truth and add light noise (observations \hat{x})
with torch.no_grad():
    x_true = rk4_integrate(lambda x, b, l: pendulum_rhs(x, b, l, g_true),
                           x0_batch, t_grid, beta_true, ell_true)  # [N,M,2]
noise = 0.01 * torch.randn_like(x_true)
x_obs = (x_true + noise).detach()  # \hat{x}(t_k), shape [N,M,2]

# -----------------------------
# 2) Black-box Neural ODE
# -----------------------------
# Use a neural network to learn the dynamics instead of the known pendulum equations

class NeuralODE(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        # Neural network to approximate dx/dt = f(x)
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        ).to(device)

    def forward(self, x0, t):
        # Integrate using RK4
        return rk4_integrate(lambda x: self.net(x), x0, t)  # [N,M,2]

model = NeuralODE(hidden_dim=64).to(device)

# -----------------------------
# 3) Loss and optimizer
# -----------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)
mse = nn.MSELoss()

# -----------------------------
# 4) Training loop
# -----------------------------
for epoch in range(501):
    optimizer.zero_grad()
    x_sim = model(x0_batch, t_grid)      # [N,M,2]
    loss = mse(x_sim, x_obs)             # averages over N,M,2
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"[{epoch:04d}] loss={loss.item():.6f}")

# -----------------------------
# 5) Inference / evaluation
# -----------------------------
with torch.no_grad():
    x_fit = model(x0_batch, t_grid)  # [N,M,2]

# -----------------------------
# 6) Plotting function
# -----------------------------
def plot_trajectories(t_grid, x_true, x_fit, num_trajectories=3):
    """
    Plot true vs. fitted trajectories for a few sample trajectories.
    t_grid: [N], x_true: [N,M,2], x_fit: [N,M,2]
    """
    # Convert tensors to CPU for plotting
    t_grid = t_grid.cpu().numpy()
    x_true = x_true.cpu().numpy()
    x_fit = x_fit.cpu().numpy()

    # Select a few trajectories to plot
    indices = torch.randperm(M)[:num_trajectories].numpy()

    # Create subplots for angle (u) and angular velocity (v)
    fig, axes = plt.subplots(2, num_trajectories, figsize=(5*num_trajectories, 8), sharex=True)

    for i, idx in enumerate(indices):
        # Plot angle (u)
        axes[0, i].plot(t_grid, x_true[:, idx, 0], 'b-', label='True Angle (u)')
        axes[0, i].plot(t_grid, x_fit[:, idx, 0], 'r--', label='Fitted Angle (u)')
        axes[0, i].set_xlabel('Time (s)')
        axes[0, i].set_ylabel('Angle (rad)')
        axes[0, i].set_title(f'Trajectory {idx+1}: Angle')
        axes[0, i].legend()
        axes[0, i].grid(True)

        # Plot angular velocity (v)
        axes[1, i].plot(t_grid, x_true[:, idx, 1], 'b-', label='True Angular Velocity (v)')
        axes[1, i].plot(t_grid, x_fit[:, idx, 1], 'r--', label='Fitted Angular Velocity (v)')
        axes[1, i].set_xlabel('Time (s)')
        axes[1, i].set_ylabel('Angular Velocity (rad/s)')
        axes[1, i].set_title(f'Trajectory {idx+1}: Angular Velocity')
        axes[1, i].legend()
        axes[1, i].grid(True)

    plt.tight_layout()
    plt.show()

# Call the plotting function
plot_trajectories(t_grid, x_true, x_fit, num_trajectories=3)