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
mol_start_time = time.time()
nx = 101  # Increased from 21 to 101 points
x_data = torch.linspace(0, 1, nx).view(-1, 1)
h_mol_np = RefMethod.run_mol_solution(nx=nx, tf=1.0)
mol_end_time = time.time()
mol_total_time = mol_end_time - mol_start_time
h_data = torch.tensor(h_mol_np).view(-1, 1).float()

print(f"Generated {nx} data points from MOL solution")
print(f"MOL solution time: {mol_total_time:.4f} seconds")

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

print("\nStarting PINN training...")
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

    # Log at intervals
    if epoch % log_every == 0 or epoch == 1:
        epoch_time = time.time() - epoch_start
        times.append(epoch_time)
        loss_data_hist.append(mse_data.item())
        loss_phys_hist.append(mse_phys.item())
        loss_bc_hist.append(loss_bc.item())
        loss_total_hist.append(loss.item())
        print(f"Epoch {epoch:4d} | L_data={mse_data:.2e} | L_phys={mse_phys:.2e} "
              f"| L_bc={loss_bc:.2e} | L_tot={loss:.2e} | t={epoch_time:.3f}s")

end_train = time.time()
pinn_total_time = end_train - start_train
print(f"\nTotal PINN training time: {pinn_total_time:.2f} seconds")

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

# Calculate number of equations for each method
mol_num_eqs = nx  # Number of spatial discretization points
pinn_num_eqs = len(list(model.parameters()))  # Number of neural network parameters

# Calculate actual parameter count for PINN
pinn_params = sum(p.numel() for p in model.parameters())

# -----------------------------
# Comprehensive Comparison Table
# -----------------------------
print("\n" + "="*80)
print("COMPREHENSIVE METHOD COMPARISON")
print("="*80)

# Create comparison table
comparison_data = {
    'Method': ['MOL (Method of Lines)', 'PINN (Physics-Informed NN)'],
    'Total Time (s)': [f'{mol_total_time:.4f}', f'{pinn_total_time:.2f}'],
    'Num Equations/Parameters': [mol_num_eqs, pinn_params],
    'Time per Eq/Param (ms)': [f'{(mol_total_time/mol_num_eqs)*1000:.4f}', 
                               f'{(pinn_total_time/pinn_params)*1000:.6f}'],
    'Max Absolute Error': ['N/A (Reference)', f'{final_max_error:.2e}'],
    'MSE': ['N/A (Reference)', f'{final_mse:.2e}'],
    'MAE': ['N/A (Reference)', f'{final_mae:.2e}']
}

# Print formatted table
print(f"{'Method':<25} {'Total Time (s)':<15} {'Num Eqs/Params':<15} {'Time/Eq (ms)':<15} {'Max Error':<12} {'MSE':<12} {'MAE':<12}")
print("-" * 106)
for i in range(len(comparison_data['Method'])):
    print(f"{comparison_data['Method'][i]:<25} "
          f"{comparison_data['Total Time (s)'][i]:<15} "
          f"{comparison_data['Num Equations/Parameters'][i]:<15} "
          f"{comparison_data['Time per Eq/Param (ms)'][i]:<15} "
          f"{comparison_data['Max Absolute Error'][i]:<12} "
          f"{comparison_data['MSE'][i]:<12} "
          f"{comparison_data['MAE'][i]:<12}")

print("\n" + "="*80)
print("DETAILED METRICS")
print("="*80)
print(f"MOL Solution:")
print(f"  - Spatial points: {nx}")
print(f"  - Solution time: {mol_total_time:.4f} seconds")
print(f"  - Time per spatial point: {(mol_total_time/nx)*1000:.4f} ms")

print(f"\nPINN Solution:")
print(f"  - Neural network parameters: {pinn_params}")
print(f"  - Training epochs: {epochs}")
print(f"  - Total training time: {pinn_total_time:.2f} seconds")
print(f"  - Average time per epoch: {np.mean(times):.3f} seconds")
print(f"  - Time per parameter: {(pinn_total_time/pinn_params)*1000:.6f} ms")
print(f"  - Collocation points used: {N_colloc}")

print(f"\nAccuracy Metrics (PINN vs MOL):")
print(f"  - Final MSE: {final_mse:.2e}")
print(f"  - Final MAE: {final_mae:.2e}")
print(f"  - Maximum absolute error: {final_max_error:.2e}")
print(f"  - Relative error (L2 norm): {(torch.norm(model(x_data) - h_data) / torch.norm(h_data)).item():.2e}")

print("\n" + "="*80)
print("COMPUTATIONAL EFFICIENCY SUMMARY")
print("="*80)
speedup_factor = pinn_total_time / mol_total_time
print(f"Speed comparison: PINN is {speedup_factor:.1f}x {'slower' if speedup_factor > 1 else 'faster'} than MOL")
print(f"MOL: {mol_total_time:.4f}s for {nx} points")
print(f"PINN: {pinn_total_time:.2f}s for {pinn_params} parameters")
print("="*80)

# -----------------------------
# Plotting Results (Accuracy plot removed)
# -----------------------------
epochs_logged = list(range(1, epochs+1, log_every))
if epochs_logged[-1] != epochs:
    epochs_logged.append(epochs)

# Create plots without accuracy comparison
fig = plt.figure(figsize=(12, 8))

# Loss curves
plt.subplot(2, 2, 1)
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

# Solution comparison
plt.subplot(2, 2, 2)
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

# Error plot
plt.subplot(2, 2, 3)
with torch.no_grad():
    error = torch.abs(model(x_data) - h_data)
plt.plot(x_data.numpy(), error.numpy(), 'r-', linewidth=2)
plt.xlabel('x')
plt.ylabel('Absolute Error')
plt.title('PINN Prediction Error')
plt.grid(True, alpha=0.3)
plt.yscale('log')

# Training time per epoch
plt.subplot(2, 2, 4)
plt.plot(epochs_logged, times, 'g-o', linewidth=2, markersize=4)
plt.xlabel('Epoch')
plt.ylabel('Time per Epoch (s)')
plt.title('Training Time per Epoch')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

