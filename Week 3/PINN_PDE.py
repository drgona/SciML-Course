"""
Vanilla PINN Example 4.1.2 (1D Heat Equation PDE)

"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 1. Data: Collocation, Initial, and Boundary Points ---
# Define physical parameters and domain
alpha = 0.01  # Thermal diffusivity
x_min, x_max = 0.0, 1.0
t_min, t_max = 0.0, 1.0


# Initial condition (IC) function: u(x, 0) = sin(pi*x)
def u_initial(x):
    return np.sin(np.pi * x)


# Generate collocation points (for PDE residual)
num_collocation = 5000
x_collocation = torch.rand(num_collocation, 1) * (x_max - x_min) + x_min
t_collocation = torch.rand(num_collocation, 1) * (t_max - t_min) + t_min
collocation_points = torch.cat([x_collocation, t_collocation], dim=1)
collocation_points.requires_grad = True

# Generate initial condition points (IC)
num_ic = 100
x_ic = torch.rand(num_ic, 1) * (x_max - x_min) + x_min
t_ic = torch.zeros(num_ic, 1)
ic_points = torch.cat([x_ic, t_ic], dim=1)
u_ic = torch.tensor(u_initial(x_ic.detach().numpy()), dtype=torch.float32)

# Generate boundary condition points (BC)
num_bc = 100
t_bc = torch.rand(num_bc, 1) * (t_max - t_min) + t_min
x_bc_left = torch.zeros(num_bc, 1)
x_bc_right = torch.ones(num_bc, 1)
bc_points_left = torch.cat([x_bc_left, t_bc], dim=1)
bc_points_right = torch.cat([x_bc_right, t_bc], dim=1)
u_bc = torch.zeros(2 * num_bc, 1)


# --- 2. Machine Learning Model: Neural Network ---
# This network will approximate the solution u(x, t)

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),  # 2 inputs: x and t
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)  # 1 output: u
        )

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)


pinn = PINN()


# --- 3. Domain Layer: The 1D Heat Equation PDE ---
# This function defines the PDE residual

def pde_residual(pinn_model, x, t):
    # Predict the solution u(x, t)
    u = pinn_model(x, t)

    # Compute partial derivatives using autograd
    # First derivatives
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True, retain_graph=True)[0]

    # Second spatial derivative
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]

    # Calculate the PDE residual: u_t - alpha * u_xx
    pde_res = u_t - alpha * u_xx
    return pde_res


# --- 4. Physics-Informed Loss Functions ---
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)

epochs = 40000

# --- 5. Training and Results ---
print(f"Training started for {epochs} epochs...")

for epoch in range(epochs):
    optimizer.zero_grad()

    # PDE Loss
    pde_res = pde_residual(pinn, collocation_points[:, 0:1], collocation_points[:, 1:2])
    pde_loss = loss_function(pde_res, torch.zeros_like(pde_res))

    # IC Loss
    u_ic_pred = pinn(ic_points[:, 0:1], ic_points[:, 1:2])
    ic_loss = loss_function(u_ic_pred, u_ic)

    # BC Loss
    u_bc_pred_left = pinn(bc_points_left[:, 0:1], bc_points_left[:, 1:2])
    u_bc_pred_right = pinn(bc_points_right[:, 0:1], bc_points_right[:, 1:2])
    bc_loss = loss_function(u_bc_pred_left, torch.zeros_like(u_bc_pred_left)) + \
              loss_function(u_bc_pred_right, torch.zeros_like(u_bc_pred_right))

    # Total loss with weights
    loss = pde_loss + 10 * ic_loss + 10 * bc_loss

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 2000 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Total Loss: {loss.item():.6f}, "
              f"PDE Loss: {pde_loss.item():.6f}, IC Loss: {ic_loss.item():.6f}, "
              f"BC Loss: {bc_loss.item():.6f}")

print("Training finished!")

# --- Visualize Final Results ---
# Generate a grid of points for plotting
n_points = 100
x_plot = np.linspace(x_min, x_max, n_points)
t_plot = np.linspace(t_min, t_max, n_points)
X_plot, T_plot = np.meshgrid(x_plot, t_plot)
x_plot_flat = torch.tensor(X_plot.flatten(), dtype=torch.float32).view(-1, 1)
t_plot_flat = torch.tensor(T_plot.flatten(), dtype=torch.float32).view(-1, 1)

with torch.no_grad():
    u_pred = pinn(x_plot_flat, t_plot_flat).detach().numpy().reshape(n_points, n_points)

# 1. 3D Plot of PINN Solution
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X_plot, T_plot, u_pred, cmap='viridis')
ax.set_title("PINN Solution for 1D Heat Equation")
ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_zlabel("u(x, t)")
plt.show()

# 2. Comparison Plot at a specific time slice
try:
    # Analytical solution for the heat equation
    def u_analytical(x, t, alpha, terms=20):
        sol = np.zeros_like(x)
        for n in range(1, terms + 1):
            term = (2 / np.pi) * (1 - (-1) ** n) / n
            sol += term * np.sin(n * np.pi * x) * np.exp(-n ** 2 * np.pi ** 2 * alpha * t)
        return sol


    t_slice = 0.5  # Choose a time to compare
    x_slice = np.linspace(x_min, x_max, n_points)
    u_analytical_slice = u_analytical(x_slice, t_slice, alpha)
    u_pinn_slice = pinn(torch.tensor(x_slice, dtype=torch.float32).view(-1, 1),
                        torch.full((n_points, 1), t_slice, dtype=torch.float32)).detach().numpy()

    plt.figure(figsize=(10, 6))
    plt.plot(x_slice, u_analytical_slice, label='Analytical Solution', color='blue', linestyle='--')
    plt.plot(x_slice, u_pinn_slice, label='PINN Prediction', color='red')
    plt.title(f"Comparison at t = {t_slice}")
    plt.xlabel("x")
    plt.ylabel("u(x, t)")
    plt.legend()
    plt.grid(True)
    plt.show()

except Exception as e:
    print(f"Could not plot analytical solution due to error: {e}")