# compare_derivatives.py
# Requires: numpy, sympy, torch (CPU is fine), optionally matplotlib for a quick plot.

import time
import numpy as np
import sympy as sp
import torch

# ----------------------------
# 1) Define function in NumPy / SymPy / PyTorch
# ----------------------------
def f_np(x):
    return np.exp(np.sin(x**2))

x_sym = sp.symbols('x')
f_sym = sp.exp(sp.sin(x_sym**2))
df_sym = sp.diff(f_sym, x_sym)                         # true derivative: e^{sin(x^2)} * cos(x^2) * 2x
f_np_true = sp.lambdify(x_sym, df_sym, modules='numpy')  # vectorized numpy function

def f_torch(x_t):
    # x_t is a 1D torch tensor
    return torch.exp(torch.sin(x_t**2))

# ----------------------------
# 2) Numerical forward-difference
#    Use a step h that scales with x to reduce cancellation (sqrt(eps) heuristic)
# ----------------------------
def forward_diff(f, x):
    eps = np.finfo(float).eps
    h = np.sqrt(eps) * (1.0 + np.abs(x))
    return (f(x + h) - f(x)) / h

# ----------------------------
# 3) Test points and ground truth
# ----------------------------
np.random.seed(0)
N = 20_000                      # number of evaluation points
x_vals = np.random.uniform(-2.0, 2.0, size=N).astype(np.float64)

truth = f_np_true(x_vals)       # symbolic ground-truth derivative (NumPy array, shape [N])

# ----------------------------
# 4) Timing helpers
# ----------------------------
def time_call(fn, *args, repeats=1, **kwargs):
    # warmup
    fn(*args, **kwargs)
    t0 = time.perf_counter()
    for _ in range(repeats):
        out = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return out, (t1 - t0) / repeats

# ----------------------------
# 5) Run: Symbolic (lookup), Numerical, AD (forward-mode JVP)
# ----------------------------

# A) "Symbolic" evaluation time (just evaluating the lambdified expression)
_, t_sym = time_call(f_np_true, x_vals, repeats=3)     # timing the numeric evaluation of the closed form

# B) Numerical forward difference
num_fd, t_num = time_call(forward_diff, f_np, x_vals, repeats=3)

# C) PyTorch forward-mode AD via JVP
#    For vector input x in R^N and elementwise f: R^N -> R^N,
#    JVP with v=ones returns elementwise df/dx.
torch.set_grad_enabled(False)  # JVP doesn't need grad recording graph here
x_t = torch.from_numpy(x_vals.copy())                  # 1D tensor [N], float64 by default?
x_t = x_t.to(dtype=torch.float64)                      # keep double precision for a fair comparison

def ad_jvp(x_t):
    v = torch.ones_like(x_t, dtype=x_t.dtype)
    # jvp returns (y, J v)
    y, jvp = torch.autograd.functional.jvp(f_torch, (x_t,), (v,))
    return jvp.detach().cpu().numpy()

ad_out, t_ad = time_call(ad_jvp, x_t, repeats=3)

# ----------------------------
# 6) Accuracy metrics
# ----------------------------
def metrics(approx, ref):
    abs_err = np.abs(approx - ref)
    rel_err = abs_err / (np.abs(ref) + 1e-15)
    return {
        "MAE": abs_err.mean(),
        "MaxAE": abs_err.max(),
        "MedAE": np.median(abs_err),
        "RMSE": np.sqrt((abs_err**2).mean()),
        "MedRel": np.median(rel_err),
        "MaxRel": rel_err.max(),
    }

m_num = metrics(num_fd, truth)
m_ad  = metrics(ad_out, truth)

# ----------------------------
# 7) Report
# ----------------------------
def fmt_time(t):
    if t < 1e-3:
        return f"{t*1e6:.1f} µs"
    if t < 1.0:
        return f"{t*1e3:.2f} ms"
    return f"{t:.3f} s"

print("\n=== Derivative of f(x) = exp(sin(x^2)) at N = {:,} points ===".format(N))
print("Ground truth (symbolic) eval time:   {:>10}".format(fmt_time(t_sym)))
print("Numerical forward diff time:         {:>10}".format(fmt_time(t_num)))
print("PyTorch AD (forward JVP) time:       {:>10}".format(fmt_time(t_ad)))

def print_metrics(name, m):
    print(f"\n{name} accuracy vs. symbolic truth")
    print("  MAE   = {:.3e}".format(m["MAE"]))
    print("  RMSE  = {:.3e}".format(m["RMSE"]))
    print("  MedAE = {:.3e}".format(m["MedAE"]))
    print("  MaxAE = {:.3e}".format(m["MaxAE"]))
    print("  MedRel= {:.3e}".format(m["MedRel"]))
    print("  MaxRel= {:.3e}".format(m["MaxRel"]))

print_metrics("Forward difference", m_num)
print_metrics("PyTorch AD (JVP)",  m_ad)

# ----------------------------
# (Optional) Quick visual check on a small subset
# ----------------------------
try:
    import matplotlib.pyplot as plt
    idx = np.argsort(x_vals)[:500]   # 500 sorted points for a neat plot
    xs = x_vals[idx]
    plt.figure(figsize=(6,4))
    plt.plot(xs, truth[idx], label="Symbolic (truth)")
    plt.plot(xs, num_fd[idx], '--', label="Forward diff")
    plt.plot(xs, ad_out[idx], ':', label="AD (JVP)")
    plt.xlabel("x")
    plt.ylabel("df/dx")
    plt.legend()
    plt.tight_layout()
    plt.show()
except Exception as e:
    # plotting is optional; ignore if matplotlib is missing
    pass
