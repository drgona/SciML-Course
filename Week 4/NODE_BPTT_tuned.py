# Tuned PyTorch Neural ODE with a black-box neural network for a damped pendulum
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, else CPU
torch.manual_seed(0)  # Set random seed for reproducibility

# -----------------------------
# 1) Problem setup and data gen
# -----------------------------
# State: x = [u, v] with u = angle (rad), v = angular velocity (rad/s)
# Dynamics: du/dt = v, dv/dt = -beta * v - (g/ell) * sin(u) (but we'll learn this with a neural network)

g_true     = 9.81   # True gravity constant (m/s^2)
beta_true  = 0.25   # True damping coefficient
ell_true   = 0.9    # True pendulum length (m)

# Time grid (shared across trajectories)
T      = 5.0  # Total time (s)
N      = 200  # Number of time steps
t_grid = torch.linspace(0.0, T, N, device=device)  # Time grid [N]
h      = t_grid[1] - t_grid[0]  # Time step size

# Synthetic dataset: M trajectories with different initial conditions
M = 32  # Number of trajectories
def sample_ic(M):
    # Generate random initial conditions for angle and angular velocity
    u0 = 0.5 * (2*torch.rand(M, device=device)-1)   # Random angles ~ [-0.5, 0.5] rad
    v0 = 0.5 * (2*torch.rand(M, device=device)-1)   # Random velocities ~ [-0.5, 0.5] rad/s
    return torch.stack([u0, v0], dim=-1)  # Stack to [M, 2]

x0_batch = sample_ic(M)  # Initial conditions for M trajectories [M,2]

def pendulum_rhs(x, beta, ell, g=9.81):
    # Analytical pendulum dynamics for ground truth
    # x: [..., 2] -> returns [..., 2] (derivatives du/dt, dv/dt)
    u, v = x[..., 0], x[..., 1]
    du = v  # du/dt = v
    dv = -beta * v - (g / ell) * torch.sin(u)  # dv/dt = -beta*v - (g/ell)*sin(u)
    return torch.stack([du, dv], dim=-1)

def rk4_integrate(f, x0, t, *f_args):
    """
    Runge-Kutta 4 for a batch of trajectories.
    x0: [M,2], t: [N], returns x: [N,M,2]
    Assumes shared time grid for the whole batch.
    """
    M = x0.shape[0]  # Number of trajectories
    N = t.shape[0]   # Number of time steps
    x = torch.zeros(N, M, 2, device=x0.device, dtype=x0.dtype)  # Initialize output tensor
    x = x.clone()  # Create a fresh tensor to avoid inplace issues
    x = torch.index_copy(x, 0, torch.tensor(0, device=x0.device), x0.unsqueeze(0))  # Set initial condition

    for k in range(N-1):
        h = t[k+1] - t[k]  # Time step
        xk = x[k]  # Current state
        k1 = f(xk, *f_args)  # RK4 stage 1
        k2 = f(xk + 0.5*h*k1, *f_args)  # RK4 stage 2
        k3 = f(xk + 0.5*h*k2, *f_args)  # RK4 stage 3
        k4 = f(xk + h*k3, *f_args)  # RK4 stage 4
        # Compute next step without inplace modification
        x_next = xk + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        x = torch.index_copy(x, 0, torch.tensor(k+1, device=x0.device), x_next.unsqueeze(0))
    return x  # [N, M, 2]

# Generate noise-free ground truth and add light noise
with torch.no_grad():
    x_true = rk4_integrate(lambda x, b, l: pendulum_rhs(x, b, l, g_true),
                           x0_batch, t_grid, beta_true, ell_true)  # True trajectories [N,M,2]
noise = 0.01 * torch.randn_like(x_true)  # Add Gaussian noise
x_obs = (x_true + noise).detach()  # Observed trajectories [N,M,2]

# Normalize data to zero mean and unit variance to stabilize training
# Improvement: Data normalization helps handle different scales of u and v
x_obs_mean = x_obs.mean(dim=[0, 1], keepdim=True)  # Compute mean over time and batch
x_obs_std = x_obs.std(dim=[0, 1], keepdim=True) + 1e-6  # Compute std, add small constant to avoid division by zero
x_obs_normalized = (x_obs - x_obs_mean) / x_obs_std  # Normalize observed data
x0_batch_normalized = (x0_batch - x_obs_mean[0]) / x_obs_std[0]  # Normalize initial conditions

# -----------------------------
# 2) Black-box Neural ODE
# -----------------------------
class NeuralODE(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        # Define neural network to learn dynamics dx/dt = f(x)
        # Improvement: Increased hidden_dim to 128 for greater model capacity
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        ).to(device)
        # Improvement 2: Apply He initialization to avoid vanishing gradients and trivial dynamics
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x0, t):
        # Integrate dynamics using RK4
        return rk4_integrate(lambda x:  self.net(x), x0, t)  # [N,M,2]

model = NeuralODE(hidden_dim=128).to(device)

# -----------------------------
# 3) Loss and optimizer
# -----------------------------
# Improvement: Add weight decay for L2 regularization to prevent overfitting to trivial solutions
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
# Improvement: Add learning rate scheduler to reduce lr when loss plateaus
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, verbose=True)
mse = nn.MSELoss()

# -----------------------------
# 4) Training loop
# -----------------------------
def compute_derivative_norm(model, x_sim):
    """
    Compute the mean L2 norm of the derivatives to penalize trivial solutions.
    Improvement: Penalizing small derivatives encourages non-trivial dynamics.
    """
    with torch.enable_grad():
        x_sim = x_sim.clone().requires_grad_(True)
        dx_dt = model.net(x_sim)  # [N,M,2]
        norm = torch.mean(torch.norm(dx_dt, dim=-1))  # Mean L2 norm over batch and time
    return norm

# Improvement: Increase to 1000 epochs to allow more time for convergence
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    x_sim = model(x0_batch_normalized, t_grid)  # Simulate trajectories [N,M,2]
    # Rescale simulated trajectories to original scale for loss computation
    x_sim_rescaled = x_sim * x_obs_std + x_obs_mean
    # Compute MSE loss between simulated and observed trajectories
    mse_loss = mse(x_sim_rescaled, x_obs)
    # Improvement: Add penalty for small derivative norms to avoid trivial solutions
    deriv_norm = compute_derivative_norm(model, x_sim)
    penalty = 0.01 * (1.0 / (deriv_norm + 1e-6))  # Penalize small derivatives
    loss = mse_loss + penalty
    loss.backward()
    # Improvement: Clip gradients to stabilize training
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    # Improvement: Step scheduler to adjust learning rate based on MSE loss
    scheduler.step(mse_loss)

    # Improvement: Enhanced logging to monitor loss and derivative norm
    if epoch % 100 == 0:
        print(f"[{epoch:04d}] mse_loss={mse_loss.item():.6f}  deriv_norm={deriv_norm.item():.6f}  lr={optimizer.param_groups[0]['lr']:.6f}")

# -----------------------------
# 5) Inference / evaluation
# -----------------------------
with torch.no_grad():
    x_fit = model(x0_batch_normalized, t_grid)  # Simulate fitted trajectories [N,M,2]
    x_fit = x_fit * x_obs_std + x_obs_mean  # Rescale to original units

# -----------------------------
# 6) Plotting function
# -----------------------------
def plot_trajectories(t_grid, x_true, x_fit, num_trajectories=3):
    """
    Plot true vs. fitted trajectories for a few sample trajectories.
    t_grid: [N], x_true: [N,M,2], x_fit: [N,M,2]
    """
    t_grid = t_grid.cpu().numpy()
    x_true = x_true.cpu().numpy()
    x_fit = x_fit.cpu().numpy()
    indices = torch.randperm(M)[:num_trajectories].numpy()
    fig, axes = plt.subplots(2, num_trajectories, figsize=(5*num_trajectories, 8), sharex=True)
    for i, idx in enumerate(indices):
        axes[0, i].plot(t_grid, x_true[:, idx, 0], 'b-', label='True Angle (u)')
        axes[0, i].plot(t_grid, x_fit[:, idx, 0], 'r--', label='Fitted Angle (u)')
        axes[0, i].set_xlabel('Time (s)')
        axes[0, i].set_ylabel('Angle (rad)')
        axes[0, i].set_title(f'Trajectory {idx+1}: Angle')
        axes[0, i].legend()
        axes[0, i].grid(True)
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