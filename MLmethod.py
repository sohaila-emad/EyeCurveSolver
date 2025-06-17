import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
# -----------------------------
# Step 1: Define MOL function
# -----------------------------
import refMethod 
# -----------------------------
# Step 2: Run MOL once
# -----------------------------
h_mol_np = refMethod.run_mol_solution()
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
for epoch in range(epochs):
    optimizer.zero_grad()
    loss = loss_fn(model, x, h_mol)
    loss.backward()
    optimizer.step()
    if epoch % 200 == 0:
        acc = 1 - (torch.norm(model(x) - h_mol) / torch.norm(h_mol)).item()
        print(f"Epoch {epoch} | Loss = {loss.item():.6f} | Accuracy ≈ {acc:.4f}")

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
