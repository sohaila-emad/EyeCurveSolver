import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
# -----------------------------
# Step 1: Define MOL function
# -----------------------------
def run_mol_solution(nx=21, tf=1.0):
    R, k, P, T = 1.0, 1.0, 1.0, 1.0
    a = R**2 * k / T
    b = R * P / T
    xl = 1.0
    dx = xl / (nx - 1)
    xg = np.linspace(0, xl, nx)
    u0 = np.full(nx, 0.25)

    def dss006(xl, xu, n, u):
        dx = (xu - xl) / (n - 1)
        r6fdx = 1 / (720 * dx)
        ux = np.zeros_like(u)
        ux[0] = r6fdx * (-1764*u[0]+4320*u[1]-5400*u[2]+4800*u[3]-2700*u[4]+864*u[5]-120*u[6])
        ux[1] = r6fdx * (-120*u[0]-924*u[1]+1800*u[2]-1200*u[3]+600*u[4]-180*u[5]+24*u[6])
        ux[2] = r6fdx * (24*u[0]-288*u[1]-420*u[2]+960*u[3]-360*u[4]+96*u[5]-12*u[6])
        for i in range(3, n-3):
            ux[i] = r6fdx * (-12*u[i-3]+108*u[i-2]-540*u[i-1]+540*u[i+1]-108*u[i+2]+12*u[i+3])
        ux[n-3] = r6fdx * (-24*u[n-1]+288*u[n-2]+420*u[n-3]-960*u[n-4]+360*u[n-5]-96*u[n-6]+12*u[n-7])
        ux[n-2] = r6fdx * (120*u[n-1]+924*u[n-2]-1800*u[n-3]+1200*u[n-4]-600*u[n-5]+180*u[n-6]-24*u[n-7])
        ux[n-1] = r6fdx * (1764*u[n-1]-4320*u[n-2]+5400*u[n-3]-4800*u[n-4]+2700*u[n-5]-864*u[n-6]+120*u[n-7])
        return ux

    def dss046(xl, xu, n, u, ux, nl, nu):
        dx = (xu - xl) / (n - 1)
        rdxs = 1 / dx**2
        uxx = np.zeros_like(u)
        if nl == 2:
            uxx[0] = rdxs * (-7.49388888888886*u[0]+12*u[1]-7.5*u[2]+4.44444444444457*u[3]-1.875*u[4]+0.48*u[5]-0.055555555555568*u[6]-4.9*ux[0]*dx)
        uxx[1] = rdxs * (0.7*u[0]-0.388888888888889*u[1]-2.7*u[2]+4.75*u[3]-3.72222222222222*u[4]+1.8*u[5]-0.5*u[6]+0.061111111111111*u[7])
        uxx[2] = rdxs * (-0.061111111111111*u[0]+1.18888888888889*u[1]-2.1*u[2]+0.722222222222223*u[3]+0.472222222222222*u[4]-0.3*u[5]+0.088888888888889*u[6]-0.011111111111111*u[7])
        for i in range(3, n-3):
            uxx[i] = rdxs * (0.011111111111111*u[i-3]-0.15*u[i-2]+1.5*u[i-1]+1.5*u[i+1]-0.15*u[i+2]+0.011111111111111*u[i+3]-2.72222222222222*u[i])
        uxx[n-3] = rdxs * (-0.061111111111111*u[n-1]+1.18888888888889*u[n-2]-2.1*u[n-3]+0.722222222222223*u[n-4]+0.472222222222222*u[n-5]-0.3*u[n-6]+0.088888888888889*u[n-7]-0.011111111111111*u[n-8])
        uxx[n-2] = rdxs * (0.7*u[n-1]-0.388888888888889*u[n-2]-2.7*u[n-3]+4.75*u[n-4]-3.72222222222222*u[n-5]+1.8*u[n-6]-0.5*u[n-7]+0.061111111111111*u[n-8])
        if nu == 1:
            uxx[n-1] = rdxs * (5.21111111111111*u[n-1]-22.3*u[n-2]+43.95*u[n-3]-52.7222222222222*u[n-4]+41*u[n-5]-20.1*u[n-6]+5.66111111111111*u[n-7]-0.7*u[n-8])
        return uxx

    def corneal_1(t, u):
        u[-1] = 0
        ux = dss006(0, xl, nx, u)
        ux[0] = 0
        uxx = dss046(0, xl, nx, u, ux, nl=2, nu=1)
        sr = np.sqrt(1 + ux**2)
        ut = uxx / sr - a * u + b / sr
        ut[-1] = 0
        return ut

    sol = solve_ivp(corneal_1, [0, tf], u0, t_eval=[tf], method='LSODA', rtol=1e-8, atol=1e-8)
    return sol.y[:, -1]

# -----------------------------
# Step 2: Run MOL once
# -----------------------------
h_mol_np = run_mol_solution()
x = torch.linspace(0, 1, 21).view(-1, 1).float()
h_mol = torch.tensor(h_mol_np).view(-1, 1).float()

# -----------------------------
# Step 3: Define PINN Model
# -----------------------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )
    def forward(self, x):
        return self.hidden(x)

model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# Step 4: Train PINN vs MOL
# -----------------------------
def loss_fn(model, x, h_true):
    h_pred = model(x)
    return torch.mean((h_pred - h_true)**2)

epochs = 2000
start_time = time.time()  # Start timing

for epoch in range(epochs):
    optimizer.zero_grad()
    loss = loss_fn(model, x, h_mol)
    loss.backward()
    optimizer.step()
    if epoch % 200 == 0:
        acc = 1 - (torch.norm(model(x) - h_mol) / torch.norm(h_mol)).item()
        print(f"Epoch {epoch} | Loss = {loss.item():.6f} | Accuracy ≈ {acc:.4f}")

end_time = time.time()  # End timing
print(f"\nTraining completed in {end_time - start_time:.2f} seconds")

# -----------------------------
# Step 5: Plot Results
# -----------------------------
x_test = torch.linspace(0, 1, 100).view(-1, 1)
h_pred_test = model(x_test).detach().numpy()

plt.plot(x.numpy(), h_mol.numpy(), 'ro', label='MOL Solution')
plt.plot(x_test.numpy(), h_pred_test, 'b-', label='PINN Prediction')
plt.xlabel("x")
plt.ylabel("h(x)")
plt.title("PINN vs MOL (Integrated)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
