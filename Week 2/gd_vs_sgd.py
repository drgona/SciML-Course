# gd_vs_sgd_true_finitesum_rosenbrock.py
# Compares full-batch Gradient Descent (GD) vs  Stochastic Gradient Descent (SGD)
# on a finite-sum nonconvex objective: average of shifted Rosenbrock components.
# Saves an animated GIF: gd_vs_sgd_true_finitesum.gif

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ---------------------- Finite-sum nonconvex objective ----------------------
# Component: shifted Rosenbrock
# f_i(x,y) = (a_i - x)^2 + beta * (y - x**2)**2
# grad_x = -2(a_i - x) - 4*beta*x*(y - x**2)
# grad_y =  2*beta*(y - x**2)

def f_i(xy, a_i, beta):
    x, y = xy
    return (a_i - x)**2 + beta * (y - x**2)**2

def grad_f_i(xy, a_i, beta):
    x, y = xy
    gx = -2*(a_i - x) - 4*beta*x*(y - x**2)
    gy =  2*beta*(y - x**2)
    return np.array([gx, gy], dtype=float)

def f_full(xy, A, beta):
    # Average over components
    return np.mean([(f_i(xy, a_i, beta)) for a_i in A])

def grad_f_full(xy, A, beta):
    g = np.zeros(2, dtype=float)
    for a_i in A:
        g += grad_f_i(xy, a_i, beta)
    return g / len(A)

# ---------------------- Problem + optimizer settings ----------------------
rng = np.random.default_rng(0)
N = 40                         # number of components in the finite sum
A = rng.uniform(-1.0, 1.0, size=N)  # shifts a_i
beta = 50.0                    # Rosenbrock parameter (nonconvexity)

iters = 220                    # animation frames / iterations
alpha_gd = 2.0e-3              # GD step size (full gradient)
alpha_sgd = 5.0e-3             # SGD step size (single-sample)
x0 = np.array([-1.7, 2.2])     # common start

# Optional: mini-batch size for "SGD"
mini_batch = 1                 # set >1 for mini-batch SGD (1 = true SGD)

# Gradient clipping to avoid blow-ups in the animation (for viz stability)
def clip(g, max_norm=10.0):
    n = np.linalg.norm(g)
    return g * (max_norm / n) if (n > max_norm and n > 0) else g

# ---------------------- Run GD (full gradient) ----------------------
gd_path = np.zeros((iters+1, 2))
gd_path[0] = x0
for t in range(1, iters+1):
    g = grad_f_full(gd_path[t-1], A, beta)
    gd_path[t] = gd_path[t-1] - alpha_gd * clip(g)

# ---------------------- Run SGD (sample i_k each step) ----------------------
sgd_path = np.zeros((iters+1, 2))
sgd_path[0] = x0
for t in range(1, iters+1):
    if mini_batch == 1:
        i_k = rng.integers(0, N)
        g = grad_f_i(sgd_path[t-1], A[i_k], beta)
    else:
        idx = rng.integers(0, N, size=mini_batch)
        g = np.mean([grad_f_i(sgd_path[t-1], A[j], beta) for j in idx], axis=0)
    sgd_path[t] = sgd_path[t-1] - alpha_sgd * clip(g)

# ---------------------- Background: contours of full objective ----------------------
# (Keep grid modest so GIF renders quickly.)
x = np.linspace(-2.5, 2.5, 220)
y = np.linspace(-0.5, 4.0, 220)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X, dtype=float)
# Evaluate average f on the grid (vectorized over (X,Y), loop over components)
for a_i in A:
    Z += (a_i - X)**2 + beta * (Y - X**2)**2
Z /= N

# ---------------------- Animation ----------------------
fig, ax = plt.subplots(figsize=(6.2, 6.2))
ax.contour(X, Y, Z, levels=25)
ax.set_xlim(x.min(), x.max())
ax.set_ylim(y.min(), y.max())
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("GD (full gradient) vs true SGD (single-sample) on a finite-sum nonconvex objective")

(gd_line,) = ax.plot([], [], label="GD path")
(sgd_line,) = ax.plot([], [], label="SGD path")
gd_point = ax.scatter([], [], marker="o", s=28, label="GD")
sgd_point = ax.scatter([], [], marker="s", s=28, label="SGD")
ax.legend(loc="upper right")

txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

def init():
    gd_line.set_data([], [])
    sgd_line.set_data([], [])
    empty = np.empty((0, 2))
    gd_point.set_offsets(empty)
    sgd_point.set_offsets(empty)
    txt.set_text("")
    return gd_line, sgd_line, gd_point, sgd_point, txt

def update(frame):
    gxy = gd_path[:frame+1]
    sxy = sgd_path[:frame+1]
    gd_line.set_data(gxy[:, 0], gxy[:, 1])
    sgd_line.set_data(sxy[:, 0], sxy[:, 1])
    gd_point.set_offsets(gxy[-1][None, :])
    sgd_point.set_offsets(sxy[-1][None, :])

    txt.set_text(
        f"iter: {frame:03d}\n"
        f"f(GD)  = {f_full(gxy[-1], A, beta):8.4f}\n"
        f"f(SGD) = {f_full(sxy[-1], A, beta):8.4f}"
    )
    return gd_line, sgd_line, gd_point, sgd_point, txt

anim = FuncAnimation(fig, update, frames=iters+1, init_func=init, blit=True, interval=45)
anim.save("gd_vs_sgd_true_finitesum.gif", writer=PillowWriter(fps=18))
print("Saved to gd_vs_sgd_true_finitesum.gif")
