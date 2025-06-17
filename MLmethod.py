import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
import RefMethod

# Generate MOL data with more points to reduce overfitting
print("Generating MOL solution with higher resolution...")
nx = 101  # Increased from 21 to 101 points
x_data = torch.linspace(0, 1, nx).view(-1, 1)
h_mol_np = run_mol_solution(nx=nx, tf=1.0)
h_data = torch.tensor(h_mol_np).view(-1, 1).float()

print(f"Generated {nx} data points from MOL solution")

# PINN model definition
a = 1.0**2 * 1.0 / 1.0
b = 1.0 * 1.0 / 1.0

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),  # Added extra layer for better capacity
            nn.Linear(50, 1)
        )

    def forward(self, x):
        return self.net(x)

model = PINN()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Collocation points - also increased for better physics coverage
N_colloc = 2000  # Increased from 1000
x_colloc = torch.rand(N_colloc, 1, requires_grad=True)

print(f"Using {N_colloc} collocation points for physics loss")

# PDE residual function
def pde_residual(x):
    u = model(x)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    sr = torch.sqrt(1 + u_x**2)
    return u_xx/sr - a*u + b/sr

# Training parameters
epochs = 2000
log_every = 100

# Metrics containers
times = []
loss_data_hist = []
loss_phys_hist = []
loss_bc_hist = []
loss_total_hist = []
accuracy_hist = []

print("\nStarting training...")
start_train = time.time()

for epoch in range(1, epochs+1):
    epoch_start = time.time()

    optimizer.zero_grad()

    # Data loss
    u_data = model(x_data)
    mse_data = torch.mean((u_data - h_data)**2)

    # Physics loss
    u_r = model(x_colloc)
    u_x = torch.autograd.grad(u_r, x_colloc, grad_outputs=torch.ones_like(u_r),
                              create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_colloc, grad_outputs=torch.ones_like(u_x),
                               create_graph=True)[0]
    sr = torch.sqrt(1 + u_x**2)
    res = u_xx/sr - a*u_r + b/sr
    mse_phys = torch.mean(res**2)

    # Boundary losses
    # u(1) = 0
    u1 = model(torch.tensor([[1.0]]))
    # u'(0) = 0
    x0 = torch.tensor([[0.0]], requires_grad=True)
    u0 = model(x0)
    u0_x = torch.autograd.grad(u0, x0, grad_outputs=torch.ones_like(u0),
                               create_graph=True)[0]
    loss_bc = u1.pow(2).mean() + u0_x.pow(2).mean()

    # Total loss with adjusted weights
    loss = mse_data + 1e-3*mse_phys + 10*loss_bc
    loss.backward()
    optimizer.step()

    # Compute accuracy wrt MOL data
    with torch.no_grad():
        acc = 1 - (torch.norm(model(x_data) - h_data) / torch.norm(h_data)).item()

    # Log at intervals
    if epoch % log_every == 0 or epoch == 1:
        epoch_time = time.time() - epoch_start
        times.append(epoch_time)
        loss_data_hist.append(mse_data.item())
        loss_phys_hist.append(mse_phys.item())
        loss_bc_hist.append(loss_bc.item())
        loss_total_hist.append(loss.item())
        accuracy_hist.append(acc)
        print(f"Epoch {epoch:4d} | L_data={mse_data:.2e} | L_phys={mse_phys:.2e} "
              f"| L_bc={loss_bc:.2e} | L_tot={loss:.2e} | Acc={acc:.4f} | t={epoch_time:.3f}s")

end_train = time.time()
print(f"\nTotal training time: {end_train - start_train:.2f} seconds")

# Final evaluation
print("\nFinal evaluation:")
with torch.no_grad():
    final_pred = model(x_data)
    final_mse = torch.mean((final_pred - h_data)**2).item()
    final_mae = torch.mean(torch.abs(final_pred - h_data)).item()
    final_max_error = torch.max(torch.abs(final_pred - h_data)).item()
    print(f"Final MSE: {final_mse:.2e}")
    print(f"Final MAE: {final_mae:.2e}")
    print(f"Max absolute error: {final_max_error:.2e}")

# -----------------------------
# Plotting Results
# -----------------------------
epochs_logged = list(range(1, epochs+1, log_every))
if epochs_logged[-1] != epochs:
    epochs_logged.append(epochs)

# Create comprehensive plots
fig = plt.figure(figsize=(15, 10))

# Loss curves
plt.subplot(2, 3, 1)
plt.plot(epochs_logged, loss_data_hist, label='Data Loss', linewidth=2)
plt.plot(epochs_logged, loss_phys_hist, label='Physics Loss', linewidth=2)
plt.plot(epochs_logged, loss_bc_hist, label='BC Loss', linewidth=2)
plt.plot(epochs_logged, loss_total_hist, label='Total Loss', linewidth=3, alpha=0.8)
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss (log scale)')
plt.legend()
plt.title('Training Loss Breakdown')
plt.grid(True, alpha=0.3)

# Accuracy curve
plt.subplot(2, 3, 2)
plt.plot(epochs_logged, accuracy_hist, 'o-', label='Accuracy', linewidth=2, markersize=4)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.legend()
plt.title('PINN Accuracy vs. MOL')
plt.grid(True, alpha=0.3)

# Solution comparison
plt.subplot(2, 3, 3)
x_test = torch.linspace(0, 1, 200).view(-1, 1)
with torch.no_grad():
    h_pred_test = model(x_test)
    h_pred_data = model(x_data)

plt.plot(x_data.numpy(), h_data.numpy(), 'ro', label=f'MOL data ({nx} pts)', markersize=3)
plt.plot(x_test.numpy(), h_pred_test.numpy(), 'b-', label='PINN prediction', linewidth=2)
plt.xlabel('x')
plt.ylabel('h(x)')
plt.title('PINN vs MOL Solution')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Summary statistics
print("\n" + "="*50)
print("TRAINING SUMMARY")
print("="*50)
print(f"Data points (MOL): {nx}")
print(f"Collocation points: {N_colloc}")
print(f"Training epochs: {epochs}")
print(f"Final accuracy: {accuracy_hist[-1]:.4f}")
print(f"Best accuracy: {max(accuracy_hist):.4f}")
print(f"Average time per epoch: {np.mean(times):.3f}s")
print("="*50)
