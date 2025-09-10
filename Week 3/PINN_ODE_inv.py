"""
Inverse PINN for the Damped Pendulum ODE:
Estimate the damping coefficient beta from measurement data.

Changes vs. vanilla:
- beta_hat is a learnable nn.Parameter.
- Adds data misfit loss using (t_meas, u_meas).
- Keeps physics (collocation) + IC losses.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import io

# -------------------------------
# 0) Config & Utilities
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Physical constants (g, l) are assumed known
g = 9.81
l = 1.0

# True beta for synthetic demo (ignored if you provide your own data)
beta_true = 0.5

# Initial conditions
u0 = np.pi / 2
v0 = 0.0

# Time domain
t_min, t_max = 0.0, 10.0

# Collocation points (physics)
num_collocation_points = 500
collocation_points = torch.linspace(t_min, t_max, num_collocation_points, requires_grad=True).view(-1, 1).to(device)

# IC tensors
t_ic = torch.tensor([[0.0]], requires_grad=True, device=device)
u_ic = torch.tensor([[u0]], dtype=torch.float32, device=device)
v_ic = torch.tensor([[v0]], dtype=torch.float32, device=device)

# Loss weights
w_phys, w_ic, w_data = 1.0, 1.0, 5.0  # increase data weight for inverse tasks (tune as needed)

# Measurement data options
use_synthetic_measurements = True     # set False if you will provide your own (t_meas_np, u_meas_np)
num_meas = 50
meas_noise_std = 0.02                 # std dev of Gaussian noise for synthetic demo

# Training
epochs = 20000
lr = 1e-3
print_every = 200

# GIF settings
make_gif = True
gif_every = 200
gif_frames = []
t_test = torch.linspace(t_min, t_max, 500).view(-1, 1).to(device)

# --------------------------------
# 1) Reference solver (for demo/plot)
# --------------------------------
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
        args=(beta_true, g, l),
        dense_output=True
    )
    t_ref = np.linspace(t_min, t_max, 500)
    u_ref = sol.sol(t_ref)[0]
except ImportError:
    sol, u_ref = None, None
    print("SciPy not found. Skipping numerical reference.")

# --------------------------------
# 2) Measurement data (provide or synthesize)
# --------------------------------
if use_synthetic_measurements:
    rng = np.random.default_rng(0)
    t_meas_np = np.sort(rng.uniform(t_min, t_max, size=num_meas))
    if sol is not None:
        u_clean = sol.sol(t_meas_np)[0]
    else:
        # fallback: small subset from reference-less case (not ideal)
        t_meas_np = np.linspace(t_min, t_max, num_meas)
        u_clean = np.interp(t_meas_np, t_ref, u_ref) if u_ref is not None else np.zeros_like(t_meas_np)
    u_meas_np = u_clean + rng.normal(0.0, meas_noise_std, size=num_meas)
else:
    # If you have real measurements, set these:
    # t_meas_np = np.array([...], dtype=float)
    # u_meas_np = np.array([...], dtype=float)
    raise ValueError("Set use_synthetic_measurements=True or provide your own measurements.")

t_meas = torch.tensor(t_meas_np, dtype=torch.float32, device=device).view(-1, 1).detach()
u_meas = torch.tensor(u_meas_np, dtype=torch.float32, device=device).view(-1, 1).detach()

# --------------------------------
# 3) PINN model
# --------------------------------
class PINN(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 1)
        )
        # Learnable beta (initialized reasonably)
        self.beta_hat = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, t):
        return self.net(t)

pinn = PINN(width=32).to(device)

# --------------------------------
# 4) Physics residual
# --------------------------------
def physics_residual(pinn_model, t, g, l):
    u = pinn_model(t)

    u_t = torch.autograd.grad(
        u, t, grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )[0]

    u_tt = torch.autograd.grad(
        u_t, t, grad_outputs=torch.ones_like(u_t),
        create_graph=True, retain_graph=True
    )[0]

    beta = pinn_model.beta_hat
    ode_res = u_tt + beta * u_t + (g / l) * torch.sin(u)
    return ode_res

# --------------------------------
# 5) Training
# --------------------------------
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(pinn.parameters(), lr=lr)

def save_frame(epoch):
    with torch.no_grad():
        plt.figure(figsize=(10, 6))

        # Current PINN prediction
        u_pred = pinn(t_test).cpu().numpy()
        plt.plot(t_test.cpu().numpy(), u_pred, label='PINN Prediction', linewidth=2)

        # Measurements
        plt.scatter(t_meas.cpu().numpy(), u_meas.cpu().numpy(), s=20, label='Measurements', alpha=0.7)

        # Reference solution (if available)
        if u_ref is not None:
            plt.plot(t_ref, u_ref, '--', label='Numerical (SciPy)')

        plt.title(f"Inverse PINN | Epoch {epoch} | beta_hat={pinn.beta_hat.item():.4f}")
        plt.xlabel("Time t")
        plt.ylabel("u(t)")
        plt.legend()
        plt.grid(True)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        gif_frames.append(Image.open(buf))
        plt.close()

print(f"Training started for {epochs} epochs...")
for epoch in range(1, epochs + 1):
    optimizer.zero_grad()

    # Physics loss on collocation points
    ode_residual = physics_residual(pinn, collocation_points, g, l)
    physics_loss = loss_fn(ode_residual, torch.zeros_like(ode_residual))

    # IC loss
    u_pred_ic = pinn(t_ic)
    u_t_pred_ic = torch.autograd.grad(
        u_pred_ic, t_ic, grad_outputs=torch.ones_like(u_pred_ic),
        create_graph=True, retain_graph=True
    )[0]
    ic_loss = loss_fn(u_pred_ic, u_ic) + loss_fn(u_t_pred_ic, v_ic)

    # Data misfit loss
    u_pred_meas = pinn(t_meas)  # no grad w.r.t t_meas needed; it is detached
    data_loss = loss_fn(u_pred_meas, u_meas)

    # Total loss
    total_loss = w_phys * physics_loss + w_ic * ic_loss + w_data * data_loss

    total_loss.backward()
    optimizer.step()

    if epoch % print_every == 0:
        print(f"Epoch [{epoch}/{epochs}] "
              f"Loss={total_loss.item():.6e} | "
              f"Phys={physics_loss.item():.3e} | IC={ic_loss.item():.3e} | Data={data_loss.item():.3e} | "
              f"beta_hat={pinn.beta_hat.item():.5f}")
        if make_gif and (epoch % gif_every == 0):
            save_frame(epoch)

print("Training finished!")
print(f"Estimated beta: {pinn.beta_hat.item():.6f} (true {beta_true:.6f})")

# --------------------------------
# 6) Final plots + GIF
# --------------------------------
with torch.no_grad():
    u_pred_final = pinn(t_test).cpu().numpy()

plt.figure(figsize=(10, 6))
plt.plot(t_test.cpu().numpy(), u_pred_final, label='PINN Prediction', linewidth=2)
plt.scatter(t_meas.cpu().numpy(), u_meas.cpu().numpy(), s=25, label='Measurements', alpha=0.8)
if u_ref is not None:
    plt.plot(t_ref, u_ref, '--', label='Numerical (SciPy)')
plt.title(f"Inverse PINN Result (beta_hat={pinn.beta_hat.item():.4f})")
plt.xlabel("Time (t)")
plt.ylabel("Angular Displacement u(t)")
plt.legend()
plt.grid(True)
plt.show()

if make_gif and gif_frames:
    imageio.mimsave('inverse_pinn_training.gif', gif_frames, fps=10)
    print("GIF saved as inverse_pinn_training.gif")
