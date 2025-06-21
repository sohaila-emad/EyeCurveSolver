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
pinn_training_time = end_train - start_train
print(f"\nTotal PINN training time: {pinn_training_time:.2f} seconds")

# ===============================================
# EXECUTION TIME MEASUREMENT (NO TRAINING)
# ===============================================

print("\n" + "="*60)
print("MEASURING EXECUTION TIME ONLY (NO TRAINING)")
print("="*60)

# Switch model to evaluation mode
model.eval()

# Measure PINN execution time for prediction only
print("\nMeasuring PINN execution time...")
pinn_exec_start = time.time()
with torch.no_grad():
    # Single forward pass through all data points
    pinn_prediction = model(x_data)
pinn_exec_end = time.time()
pinn_execution_time = pinn_exec_end - pinn_exec_start

print(f"PINN execution time (inference only): {pinn_execution_time:.6f} seconds")

# Measure execution time for multiple evaluations to get better statistics
n_runs = 100
print(f"\nMeasuring PINN execution time over {n_runs} runs...")
exec_times = []

for i in range(n_runs):
    start = time.time()
    with torch.no_grad():
        _ = model(x_data)
    end = time.time()
    exec_times.append(end - start)

avg_exec_time = np.mean(exec_times)
std_exec_time = np.std(exec_times)
min_exec_time = np.min(exec_times)
max_exec_time = np.max(exec_times)

print(f"Average execution time: {avg_exec_time:.6f} ± {std_exec_time:.6f} seconds")
print(f"Min execution time: {min_exec_time:.6f} seconds")
print(f"Max execution time: {max_exec_time:.6f} seconds")

# Final evaluation (for accuracy metrics)
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
pinn_params = sum(p.numel() for p in model.parameters())

# -----------------------------
# EXECUTION TIME COMPARISON TABLE
# -----------------------------
print("\n" + "="*80)
print("EXECUTION TIME COMPARISON (NO TRAINING)")
print("="*80)

# Create execution comparison table
exec_comparison_data = {
    'Method': ['MOL (Method of Lines)', 'PINN (Inference Only)'],
    'Execution Time (s)': [f'{mol_total_time:.6f}', f'{avg_exec_time:.6f}'],
    'Num Points/Parameters': [mol_num_eqs, pinn_params],
    'Time per Point/Param (μs)': [f'{(mol_total_time/mol_num_eqs)*1e6:.2f}', 
                                  f'{(avg_exec_time/pinn_params)*1e6:.4f}'],
    'Speedup Factor': ['1.0x (Reference)', f'{mol_total_time/avg_exec_time:.1f}x']
}

# Print formatted table
print(f"{'Method':<25} {'Exec Time (s)':<15} {'Num Points/Params':<18} {'Time/Point (μs)':<17} {'Speedup':<10}")
print("-" * 85)
for i in range(len(exec_comparison_data['Method'])):
    print(f"{exec_comparison_data['Method'][i]:<25} "
          f"{exec_comparison_data['Execution Time (s)'][i]:<15} "
          f"{exec_comparison_data['Num Points/Parameters'][i]:<18} "
          f"{exec_comparison_data['Time per Point/Param (μs)'][i]:<17} "
          f"{exec_comparison_data['Speedup Factor'][i]:<10}")

print("\n" + "="*80)
print("EXECUTION TIME SUMMARY")
print("="*80)
speedup_factor = mol_total_time / avg_exec_time
print(f"MOL execution time: {mol_total_time:.6f} seconds")
print(f"PINN execution time: {avg_exec_time:.6f} ± {std_exec_time:.6f} seconds")
print(f"Speedup factor: PINN is {speedup_factor:.1f}x {'faster' if speedup_factor > 1 else 'slower'} than MOL")
print(f"PINN processes {nx} points in {avg_exec_time*1000:.3f} milliseconds")
print(f"Time per prediction: {(avg_exec_time/nx)*1e6:.2f} microseconds per point")

print("\n" + "="*80)
print("DETAILED EXECUTION METRICS")
print("="*80)
print(f"MOL Solution:")
print(f"  - Spatial points: {nx}")
print(f"  - Total execution time: {mol_total_time:.6f} seconds")
print(f"  - Time per spatial point: {(mol_total_time/nx)*1e6:.2f} microseconds")

print(f"\nPINN Inference:")
print(f"  - Neural network parameters: {pinn_params}")
print(f"  - Input points: {nx}")
print(f"  - Average execution time: {avg_exec_time:.6f} seconds")
print(f"  - Standard deviation: {std_exec_time:.6f} seconds")
print(f"  - Time per input point: {(avg_exec_time/nx)*1e6:.2f} microseconds")
print(f"  - Time per parameter: {(avg_exec_time/pinn_params)*1e6:.4f} microseconds")

print(f"\nAccuracy Metrics (PINN vs MOL):")
print(f"  - Final MSE: {final_mse:.2e}")
print(f"  - Final MAE: {final_mae:.2e}")
print(f"  - Maximum absolute error: {final_max_error:.2e}")
print("="*80)

# -----------------------------
# h(x) Values Template at Specific Points
# -----------------------------
print("\n" + "="*80)
print("h(x) VALUES AT SPECIFIC POINTS")
print("="*80)

# Define evaluation points
eval_points = torch.tensor([[0.0], [0.2], [0.4], [0.6], [0.8], [1.0]])

# Get MOL reference values at these points (interpolated)
mol_h_values = []
pinn_h_values = []

# Measure execution time for specific point evaluation
eval_exec_start = time.time()
with torch.no_grad():
    # Get PINN predictions at evaluation points
    pinn_pred_eval = model(eval_points)
eval_exec_end = time.time()
eval_exec_time = eval_exec_end - eval_exec_start

# Interpolate MOL values at evaluation points
x_data_np = x_data.numpy().flatten()
h_data_np = h_data.numpy().flatten()

for point in eval_points.numpy().flatten():
    # Find closest indices for interpolation
    if point in x_data_np:
        # Exact match
        idx = np.where(x_data_np == point)[0][0]
        mol_value = h_data_np[idx]
    else:
        # Linear interpolation
        mol_value = np.interp(point, x_data_np, h_data_np)
    mol_h_values.append(mol_value)

pinn_h_values = pinn_pred_eval.numpy().flatten()

# Create template table
print(f"{'x':<6} {'MOL h(x)':<12} {'PINN h(x)':<12} {'Absolute Error':<15} {'Relative Error (%)':<18}")
print("-" * 65)

for i, x_val in enumerate([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]):
    mol_val = mol_h_values[i]
    pinn_val = pinn_h_values[i]
    abs_error = abs(pinn_val - mol_val)
    rel_error = (abs_error / abs(mol_val)) * 100 if mol_val != 0 else 0
    
    print(f"{x_val:<6.1f} {mol_val:<12.6f} {pinn_val:<12.6f} {abs_error:<15.6f} {rel_error:<18.4f}")

print(f"\nExecution time for 6 specific points: {eval_exec_time:.6f} seconds")
print(f"Time per specific point evaluation: {(eval_exec_time/6)*1e6:.2f} microseconds")

# -----------------------------
# Plotting Results (Execution time focused)
# -----------------------------
epochs_logged = list(range(1, epochs+1, log_every))
if epochs_logged[-1] != epochs:
    epochs_logged.append(epochs)

# Create plots focused on execution performance
fig = plt.figure(figsize=(12, 8))

# Execution time comparison
plt.subplot(2, 2, 1)
methods = ['MOL', 'PINN\n(Inference)']
exec_times_plot = [mol_total_time, avg_exec_time]
colors = ['blue', 'red']
bars = plt.bar(methods, exec_times_plot, color=colors, alpha=0.7)
plt.ylabel('Execution Time (s)')
plt.title('Execution Time Comparison')
plt.yscale('log')
for i, (bar, time_val) in enumerate(zip(bars, exec_times_plot)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1, 
             f'{time_val:.6f}s', ha='center', va='bottom')
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

# Execution time statistics
plt.subplot(2, 2, 3)
plt.hist(exec_times, bins=20, alpha=0.7, color='green', edgecolor='black')
plt.axvline(avg_exec_time, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_exec_time:.6f}s')
plt.xlabel('Execution Time (s)')
plt.ylabel('Frequency')
plt.title(f'PINN Execution Time Distribution ({n_runs} runs)')
plt.legend()
plt.grid(True, alpha=0.3)

# Error plot
plt.subplot(2, 2, 4)
with torch.no_grad():
    error = torch.abs(model(x_data) - h_data)
plt.plot(x_data.numpy(), error.numpy(), 'r-', linewidth=2)
plt.xlabel('x')
plt.ylabel('Absolute Error')
plt.title('PINN Prediction Error')
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.tight_layout()
plt.show()
