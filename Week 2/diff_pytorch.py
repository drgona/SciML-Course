import torch
import time

"""
This script provides runnable PyTorch code examples, illustrating the core concepts of Automatic
Differentiation (AD).

The examples cover:
1. Differentiating a simple function using forward-mode (JVP) and
   reverse-mode (VJP) AD.
2. Differentiating through control flow (if/else and loops).
3. Computing higher-order derivatives (Hessian-vector products).

These examples highlight PyTorch's "define-by-run" AD system, where the
computational graph is built dynamically as the code executes.
"""

# --- Start the timer ---
start_time = time.perf_counter()


# --- Example 1: Differentiating a Simple Function (VJP and JVP) ---
# Corresponds to your "Mechanics of Automatic Differentiation" section.
# PyTorch's primary API is reverse-mode (`.backward()`). Forward mode
# requires a more specific utility function.

print("--- Example 1: VJP and JVP on a simple function ---")

# The function to differentiate, y = exp(sin(x^2))
def f_simple(x):
    u1 = x**2
    u2 = torch.sin(u1)
    y = torch.exp(u2)
    return y

# Define the input value. `requires_grad=True` is crucial; it tells PyTorch
# to build the computational graph for this tensor.
x_val = torch.tensor(2.0, requires_grad=True)

# --- Reverse-mode AD (VJP) ---
# PyTorch's main reverse-mode API is `.backward()`. It computes the
# gradient of a scalar tensor with respect to all its leaf nodes.
y_val = f_simple(x_val)
y_val.backward(retain_graph=True) # Modified to retain the graph
# The gradient is stored in the `.grad` attribute of the input tensor.
dy_dx = x_val.grad
print(f"1a. Using y.backward() (Reverse-mode AD) at x={x_val.item()}: {dy_dx.item()}")

# For a more explicit VJP (vector-Jacobian product), you can use `torch.autograd.grad`.
# Note: You still need to clear the old gradient before the next computation.
x_val.grad = None
dy_dx_vjp = torch.autograd.grad(y_val, x_val, retain_graph=True)[0] # Modified to retain the graph
print(f"1b. Using torch.autograd.grad (Explicit Reverse-mode AD) at x={x_val.item()}: {dy_dx_vjp.item()}")


# --- Forward-mode AD (JVP) ---
# PyTorch uses `torch.autograd.functional.jvp` for forward-mode AD.
# It requires the function, the inputs, and the tangents.
x_val_jvp = torch.tensor(2.0, requires_grad=True)
x_dot = torch.tensor(1.0)  # The seed direction
y_val_jvp, y_dot = torch.autograd.functional.jvp(f_simple, (x_val_jvp,), (x_dot,))
print(f"1c. Using torch.autograd.functional.jvp (Forward-mode AD) at x={x_val_jvp.item()}: {y_dot.item()}")


# --- Example 2: Differentiating through Control Flow ---
# PyTorch's "define-by-run" graph handles control flow naturally, just like JAX.
# The graph is built dynamically based on the path taken by the program.

print("\n--- Example 2: Differentiating through Control Flow ---")

# A function with an if/else branch
def f_branch(x):
    if x > 0:
        return x**2
    else:
        return torch.exp(x)

# Differentiating the path where x > 0
x_positive = torch.tensor(2.0, requires_grad=True)
y_positive = f_branch(x_positive)
y_positive.backward()
grad_positive = x_positive.grad
print(f"2a. Gradient at x={x_positive.item()} (x > 0 branch): {grad_positive.item()}")

# Differentiating the path where x <= 0
x_negative = torch.tensor(-1.0, requires_grad=True)
y_negative = f_branch(x_negative)
y_negative.backward()
grad_negative = x_negative.grad
print(f"2b. Gradient at x={x_negative.item()} (x <= 0 branch): {grad_negative.item()}")


print("\n--- Example 3: Differentiating through a loop ---")

# A function with a loop. This works seamlessly because PyTorch
# traces the operations executed in the loop during the forward pass.
def f_loop(x, num_iterations):
    for _ in range(num_iterations):
        x = x**2
    return x

# Differentiating the unrolled trace of the loop
x_loop = torch.tensor(2.0, requires_grad=True)
num_iterations = 3
y_loop = f_loop(x_loop, num_iterations)
y_loop.backward()
grad_loop = x_loop.grad
print(f"3a. Gradient of a {num_iterations}-iteration loop at x={x_loop.item()}: {grad_loop.item()}")


# --- Example 4: Higher-Order Derivatives ---
# This demonstrates how to compute a Hessian-vector product using PyTorch's
# forward-over-reverse approach. It's similar to JAX's composable primitives.

print("\n--- Example 4: Higher-Order Derivatives (Hessian-vector product) ---")

# A function with a vector input
def f_vector(x):
    return torch.sin(x[0]) + torch.cos(x[1])**2

# Define the input vector and the direction vector
x_vec = torch.tensor([1.0, 0.5, 2.0], requires_grad=True)
v_vec = torch.tensor([1.0, 1.0, 1.0])

# Step 1: Create a function for the gradient using reverse-mode AD.
# We need to tell `torch.autograd.grad` to create the graph for the gradient
# itself by setting `create_graph=True`.
grad_f = torch.autograd.grad(f_vector(x_vec), x_vec, create_graph=True)[0]

# Step 2: Now, we compute the forward-mode derivative of the gradient.
# This gives us the Hessian-vector product.
Hessian_vector_product = torch.autograd.grad(grad_f, x_vec, grad_outputs=v_vec)[0]

print(f"4a. Hessian-vector product at x={x_vec.tolist()} with direction v={v_vec.tolist()}: {Hessian_vector_product.tolist()}")


# --- End the timer and print the elapsed time ---
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"\n--- Script Execution Time ---")
print(f"Elapsed time: {elapsed_time:.4f} seconds")