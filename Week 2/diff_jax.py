import jax
import jax.numpy as jnp
import time

"""
This script provides runnable JAX code examples to illustrate the concepts
of Automatic Differentiation (AD) for a class on scientific machine learning.
The examples are directly tied to the lecture notes provided, covering:

1.  Differentiating a simple function using both forward-mode (JVP) and
    reverse-mode (VJP) AD.
2.  Differentiating through control flow, such as if/else branches and loops.
3.  Computing higher-order derivatives, specifically the Hessian-vector product.

All examples use JAX, a modern library for differentiable programming.
"""

# --- Start the timer ---
start_time = time.perf_counter()


# --- Example 1: Differentiating a Simple Function (VJP and JVP) ---
# Corresponds to "Mechanics of Automatic Differentiation"
# and "Forward-/Reverse-mode AD" sections in your notes.

print("--- Example 3.2.1-3: VJP and JVP on a simple function ---")

# The function to differentiate, y = exp(sin(x^2))
def f_simple(x):
  u1 = x**2
  u2 = jnp.sin(u1)
  y = jnp.exp(u2)
  return y

# Define the input value
x_val = 2.0

# --- Reverse-mode AD (VJP) ---
# JAX's `jax.grad` is the most common way to get a gradient.
# It uses reverse-mode AD (VJP) under the hood.
dy_dx = jax.grad(f_simple)(x_val)
print(f"1a. Using jax.grad (Reverse-mode AD) at x={x_val}: {dy_dx}")

# Explicitly use the lower-level `jax.vjp` primitive
# It returns the output value and a function to compute the VJP.
y_val, vjp_fun = jax.vjp(f_simple, x_val)
# We "seed" the VJP with 1.0 because the output is a scalar.
dy_dx_vjp = vjp_fun(1.0)[0]
print(f"1b. Using jax.vjp (Explicit Reverse-mode AD) at x={x_val}: {dy_dx_vjp}")

# --- Forward-mode AD (JVP) ---
# The seed direction for a scalar input is 1.0.
x_dot = 1.0
y_val, y_dot = jax.jvp(f_simple, (x_val,), (x_dot,))
print(f"1c. Using jax.jvp (Forward-mode AD) at x={x_val}: {y_dot}")


# --- Example 2: Differentiating through Control Flow ---
# Corresponds to "Evaluation Trace with Control Flow" section in your notes.
# JAX's AD operates on the "evaluation trace," the specific path of execution.

print("\n--- Example 3.2.4: Differentiating through Control Flow ---")

# A function with an if/else branch (Example 6.6)
def f_branch(x):
  if x > 0:
    return x**2
  else:
    return jnp.exp(x)

# Differentiating the path where x > 0
x_positive = 2.0
grad_positive = jax.grad(f_branch)(x_positive)
print(f"2a. Gradient at x={x_positive} (x > 0 branch): {grad_positive}")

# Differentiating the path where x <= 0
x_negative = -1.0
grad_negative = jax.grad(f_branch)(x_negative)
print(f"2b. Gradient at x={x_negative} (x <= 0 branch): {grad_negative}")


print("\n--- Example 3.2.5: Differentiating through a loop ---")

# A function with a loop (Example 6.7)
def f_loop(x, num_iterations):
  for _ in range(num_iterations):
    x = x**2
  return x

# Differentiating the unrolled trace of the loop
x_loop = 2.0
num_iterations = 3
# We specify 'argnums=0' to differentiate with respect to the first argument (x).
grad_loop = jax.grad(f_loop, argnums=0)(x_loop, num_iterations)
print(f"3a. Gradient of a {num_iterations}-iteration loop at x={x_loop}: {grad_loop}")


# --- Example 3: Higher-Order Derivatives ---
# Corresponds to "Higher-Order Derivatives" section in your notes.
# This demonstrates "forward-over-reverse" AD to get a Hessian-vector product.

print("\n--- Example: Higher-Order Derivatives (Hessian-vector product) ---")

# A function with a vector input
def f_vector(x):
  return jnp.sin(x[0]) + jnp.cos(x[1])**2

# Define the input vector and the direction vector
x_vec = jnp.array([1.0, 0.5, 2.0])
v_vec = jnp.array([1.0, 1.0, 1.0])

# Step 1: Create a function for the gradient using reverse-mode AD.
# This gives us a function that computes the gradient vector.
grad_f = jax.grad(f_vector)

# Step 2: Apply forward-mode AD (jvp) to the gradient function.
# This computes the Jacobian-vector product of the gradient, which is H*v.
Hessian_vector_product = jax.jvp(grad_f, (x_vec,), (v_vec,))[1]

print(f"4a. Hessian-vector product at x={x_vec} with direction v={v_vec}: {Hessian_vector_product}")
# Note: The output is the directional derivative of the gradient.

# --- End the timer and print the elapsed time ---
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"\n--- Script Execution Time ---")
print(f"Elapsed time: {elapsed_time:.4f} seconds")