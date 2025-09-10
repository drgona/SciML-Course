"""
Vanilla PINN Example 4.1.1 (Damped Pendulum ODE).

Example inspired by
https://benmoseley.blog/my-research/so-what-is-a-physics-informed-neural-network/
https://www.mathworks.com/discovery/physics-informed-neural-networks.html
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import io

# --- 1. Data: Sampled Collocation Points ---
# These are the time points where we will enforce the physics.

# Define physical parameters
g = 9.81  # Acceleration due to gravity
l = 1.0  # Pendulum length
beta = 0.5  # Damping coefficient

# Initial conditions
u0 = np.pi / 2  # Initial angular displacement (radians)
v0 = 0.0  # Initial angular velocity

# Time domain
t_min, t_max = 0.0, 10.0
num_collocation_points = 500
collocation_points = torch.linspace(t_min, t_max, num_collocation_points, requires_grad=True).view(-1, 1)

# Initial condition points for loss calculation
t_ic = torch.tensor(0.0, requires_grad=True).view(-1, 1)
u_ic = torch.tensor(u0).view(-1, 1)
v_ic = torch.tensor(v0).view(-1, 1)


# --- 2. Machine Learning Model: Neural Network ---
# This network will approximate the solution u(t).

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, t):
        return self.net(t)


pinn = PINN()


# --- 3. Domain Layer: The Damped Pendulum ODE ---
# This function calculates the residual of the ODE.
# It uses PyTorch's autograd to compute derivatives.

def physics_residual(pinn_model, t):
    # Get the predicted solution u(t) from the network
    u = pinn_model(t)

    # Compute the first derivative (angular velocity)
    u_t = torch.autograd.grad(
        u, t,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True
    )[0]

    # Compute the second derivative (angular acceleration)
    u_tt = torch.autograd.grad(
        u_t, t,
        grad_outputs=torch.ones_like(u_t),
        create_graph=True,
        retain_graph=True
    )[0]

    # Define the ODE residual (should be close to zero)
    # d^2u/dt^2 + beta * du/dt + (g/l) * sin(u) = 0
    ode_residual = u_tt + beta * u_t + (g / l) * torch.sin(u)

    return ode_residual


# --- 4. GIF visualisation functions ---

# Create a list to store plot images for the GIF
gif_frames = []
t_test = torch.linspace(t_min, t_max, 500).view(-1, 1)
collocation_points_np = collocation_points.detach().numpy()

# A reference numerical solution for comparison
try:
    from scipy.integrate import solve_ivp

    def damped_pendulum_ode(t, y, beta, g, l):
        theta, omega = y
        dtheta_dt = omega
        domega_dt = -beta * omega - (g / l) * np.sin(theta)
        return [dtheta_dt, domega_dt]

    sol = solve_ivp(
        damped_pendulum_ode,
        (t_min, t_max),
        [u0, v0],
        args=(beta, g, l),
        dense_output=True
    )
    t_ref = np.linspace(t_min, t_max, 500)
    u_ref = sol.sol(t_ref)[0]
except ImportError:
    sol, u_ref = None, None
    print("SciPy not found. Skipping numerical solution in GIF.")


# plotting function for gif
def save_frame(epoch):
    """Saves a plot frame for the GIF."""
    plt.figure(figsize=(10, 6))

    # Plotting the current PINN prediction
    plt.plot(t_test.detach().numpy(), pinn(t_test).detach().numpy(),
             label='PINN Prediction', color='red', linewidth=2)

    # Plotting the collocation points
    plt.scatter(collocation_points_np, torch.zeros(collocation_points_np.shape).numpy(),
                s=10, label='Collocation Points', color='orange', alpha=0.5)

    # Plotting the numerical solution if available
    if u_ref is not None:
        plt.plot(t_ref, u_ref, label='Numerical Solution (SciPy)',
                 linestyle='--', color='blue')

    plt.title(f"PINN Training Progression | Epoch {epoch}")
    plt.xlabel("Time (t)")
    plt.ylabel("Angular Displacement u(t)")
    plt.legend()
    plt.grid(True)

    # Save the figure to an in-memory buffer without closing it
    buf = io.BytesIO()
    plt.savefig(buf, format='png')

    # Reset the buffer position to the beginning and read the image
    buf.seek(0)
    img = Image.open(buf)

    # Append the image to the list and close the figure
    gif_frames.append(img)
    plt.close()


# --- 5. Physics-Informed Loss Functions, Training and Results ---
# We combine the initial condition loss and the physics loss.

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)

epochs = 30000

print(f"Training started for {epochs} epochs...")

for epoch in range(epochs):
    optimizer.zero_grad()

    # Calculate the physics loss
    ode_residual = physics_residual(pinn, collocation_points)
    physics_loss = loss_function(ode_residual, torch.zeros_like(ode_residual))

    # Calculate the initial condition loss
    u_pred_ic = pinn(t_ic)
    u_t_pred_ic = torch.autograd.grad(
        u_pred_ic, t_ic,
        grad_outputs=torch.ones_like(u_pred_ic),
        create_graph=True,
        retain_graph=True
    )[0]

    ic_loss_pos = loss_function(u_pred_ic, u_ic)
    ic_loss_vel = loss_function(u_t_pred_ic, v_ic)
    ic_loss = ic_loss_pos + ic_loss_vel

    # Total loss is a combination of both
    total_loss = physics_loss + ic_loss

    # Backward pass and optimization
    total_loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss.item():.6f}, '
              f'Physics Loss: {physics_loss.item():.6f}, IC Loss: {ic_loss.item():.6f}')
        # Save a frame for the GIF every 100 epochs
        with torch.no_grad():
            save_frame(epoch + 1)

print("Training finished!")

# --- Visualize Final Results and Generate GIF ---
# Final plot with collocation points
with torch.no_grad():
    u_pred_final = pinn(t_test)

plt.figure(figsize=(10, 6))
plt.plot(t_test.detach().numpy(), u_pred_final.detach().numpy(), label='PINN Prediction', color='red', linewidth=2)
plt.scatter(collocation_points_np, pinn(collocation_points).detach().numpy(), s=10, label='Collocation Points',
            color='red', alpha=0.5)

if u_ref is not None:
    plt.plot(t_ref, u_ref, label='Numerical Solution (SciPy)', linestyle='--', color='blue')

plt.title("Damped Pendulum: Final PINN Solution")
plt.xlabel("Time (t)")
plt.ylabel("Angular Displacement u(t)")
plt.legend()
plt.grid(True)
plt.show()

# Generate the GIF from the collected frames
if gif_frames:
    imageio.mimsave('pinn_training_progression.gif', gif_frames, fps=20)
    print("GIF saved as pinn_training_progression.gif")