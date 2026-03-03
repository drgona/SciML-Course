import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


# Neural network policy for predicting solutions
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, xi):
        return self.network(xi)


# Self-supervised loss function with penalty for inequality constraint
def compute_self_supervised_loss(x, q, a, b, lambda_penalty=100.0):
    # Objective: ||x||_2^2 + q^T x
    quad_term = x.norm(2).pow(2)
    linear_term = q.t() @ x
    obj = quad_term + linear_term
    # Penalty for inequality constraint violation: lambda * (a^T x - b)_+^2
    constraint_violation = torch.relu(a.t() @ x - b)
    penalty = lambda_penalty * constraint_violation.pow(2)
    return obj + penalty


# L2O class for 2D QP with penalty method
class L2O_2D_QP:
    def __init__(self, input_dim, output_dim, hidden_dim=64, lambda_penalty=10.0):
        self.policy = PolicyNetwork(input_dim, output_dim, hidden_dim)
        self.lambda_penalty = lambda_penalty
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.001)

    def forward(self, xi):
        # Predict solution
        return self.policy(xi)

    def train_step(self, xi_batch, q_batch, a_batch, b_batch):
        self.optimizer.zero_grad()
        loss = 0.0
        batch_size = xi_batch.shape[0]

        for i in range(batch_size):
            xi = xi_batch[i]
            q = q_batch[i]
            a = a_batch[i]
            b = b_batch[i]

            # Forward pass
            x = self.forward(xi)
            # Compute loss
            sample_loss = compute_self_supervised_loss(x, q, a, b, self.lambda_penalty)
            loss += sample_loss

        # Average loss over batch
        loss = loss / batch_size
        loss.backward()
        self.optimizer.step()
        return loss.item()


# Visualization function for final 2D solution with constraint boundary
def visualize_2d_solution(x, q, a, b):
    # Convert tensors to numpy for plotting
    x_np = x.detach().numpy()
    q_np = q.detach().numpy()
    a_np = a.detach().numpy()
    b_np = b.detach().numpy()

    # Create grid for contour plot
    x1 = np.linspace(-2, 2, 100)
    x2 = np.linspace(-2, 2, 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros_like(X1)

    # Compute objective function over grid
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            x_grid = np.array([X1[i, j], X2[i, j]])
            Z[i, j] = np.linalg.norm(x_grid) ** 2 + q_np.T @ x_grid

    # Plot
    plt.figure(figsize=(8, 6))
    # Contour plot of objective
    plt.contour(X1, X2, Z, levels=20, cmap='viridis')
    plt.colorbar(label='Objective Value')

    # Feasible set (half-space: a^T x <= b)
    x1_line = np.linspace(-2, 2, 100)
    if abs(a_np[1]) > 1e-6:  # Avoid division by zero
        x2_line = (b_np - a_np[0] * x1_line) / a_np[1]
        # Plot feasible set
        plt.fill_between(x1_line, -2, x2_line, where=(x2_line >= -2), color='gray', alpha=0.3, label='Feasible Set')
        # Plot constraint boundary
        plt.plot(x1_line, x2_line, 'k-', label='Constraint Boundary ($a^T x = b$)')
    else:
        # If a2 is zero, constraint is a1*x1 <= b
        x1_bound = b_np / a_np[0] if abs(a_np[0]) > 1e-6 else 0
        plt.axvline(x1_bound, color='gray', alpha=0.3, label='Feasible Set')
        plt.axvline(x1_bound, color='black', label='Constraint Boundary ($a^T x = b$)')

    # Predicted solution
    plt.scatter(x_np[0], x_np[1], color='red', s=100, label='Predicted Solution')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Final 2D QP Solution with Constraint')
    plt.legend()
    plt.grid(True)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.show()


# Problem setup
xi_dim = 5  # Dimension of xi (concatenation of q(2), a(2), b(1))
n = 2  # Dimension of decision variable x (2D)
batch_size = 100
num_epochs = 1000

# Initialize model
l2o = L2O_2D_QP(input_dim=xi_dim, output_dim=n, lambda_penalty=10.0)

# Generate dummy data
xi_batch = torch.randn(batch_size, xi_dim)
q_batch = xi_batch[:, :2]  # First two components for q
a_batch = xi_batch[:, 2:4]  # Next two components for a
b_batch = xi_batch[:, 4]  # Last component for b

# Training loop
for epoch in range(num_epochs):
    loss = l2o.train_step(xi_batch, q_batch, a_batch, b_batch)
    if epoch % 20 == 0 or epoch == num_epochs - 1:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")



i = 5
# Visualize final solution for the first sample
final_x = l2o.forward(xi_batch[i])
visualize_2d_solution(final_x, q_batch[i], a_batch[i], b_batch[i])
