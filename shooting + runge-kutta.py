import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.interpolate import interp1d

start_time = time.time()

# Parameters
a = 1.0
b = 1.0
x0 = 0.0
x1 = 1.0
h_step = 0.05

# Reference Method Results (from RefMethod.py at t=1.0)
ref_x = np.linspace(0, 1, 21)
ref_h = np.array([
    0.34803, 0.34722, 0.34479, 0.34074, 0.33505,
    0.32771, 0.31871, 0.30804, 0.29566, 0.28156,
    0.26570, 0.24805, 0.22856, 0.20720, 0.18391,
    0.15864, 0.13132, 0.10188, 0.07024, 0.03632,
    0.00000
])

# Differential equation system
def f(x, y):
    y1, y2 = y
    sqrt_term = np.sqrt(1 + y2**2)
    dy1 = y2
    dy2 = (a * y1 - b / sqrt_term) * sqrt_term
    return np.array([dy1, dy2])

# Runge-Kutta 4th order step
def rk4_step(f, x, y, h):
    k1 = f(x, y)
    k2 = f(x + h/2, y + h/2 * k1)
    k3 = f(x + h/2, y + h/2 * k2)
    k4 = f(x + h, y + h * k3)
    return y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

# Integrate system from x0 to x1 given initial guess h(0) = s
def integrate(s):
    x = x0
    y = np.array([s, 0.0])  # [h, h']
    xs, hs, hs_prime = [x], [y[0]], [y[1]]
    while x < x1:
        y = rk4_step(f, x, y, h_step)
        x += h_step
        xs.append(x)
        hs.append(y[0])
        hs_prime.append(y[1])
    return xs, hs, hs_prime, y[0]

# Shooting method with residuals tracking
def shooting_method(s1, s2, tol=1e-6):
    residuals = []
    _, _, _, y1 = integrate(s1)
    _, _, _, y2 = integrate(s2)
    residuals.append(abs(y1))
    residuals.append(abs(y2))
    iteration = 0
    while abs(s2 - s1) > tol:
        s3 = s2 - y2 * (s2 - s1) / (y2 - y1)
        _, _, _, y3 = integrate(s3)
        residuals.append(abs(y3))
        s1, y1 = s2, y2
        s2, y2 = s3, y3
        iteration += 1
    return s2, residuals

# Run shooting to find correct h(0)
s_correct, residuals = shooting_method(0.1, 0.5)

# Final integration with correct h(0)
xs, hs, hs_prime, _ = integrate(s_correct)

# Interpolate reference solution to match RK4 x-points
ref_interp = interp1d(ref_x, ref_h, kind='cubic', fill_value='extrapolate')
ref_h_interp = ref_interp(xs)

# Calculate absolute and relative errors
abs_error = np.abs(np.array(hs) - ref_h_interp)
rel_error = (abs_error / np.abs(ref_h_interp)) * 100  # Percentage error

# Handle division by zero for points where ref_h is very close to zero
rel_error = np.where(np.abs(ref_h_interp) < 1e-10, 0, rel_error)

# Create comprehensive comparison table
df = pd.DataFrame({
    "x": xs,
    "h_RK4": hs,
    "h_Ref": ref_h_interp,
    "Abs_Error": abs_error,
    "Rel_Error_%": rel_error
})

end_time = time.time()
execution_time = end_time - start_time

# Statistical analysis
max_abs_error = np.max(abs_error)
max_rel_error = np.max(rel_error)
mean_abs_error = np.mean(abs_error)
mean_rel_error = np.mean(rel_error)
rms_error = np.sqrt(np.mean(abs_error**2))

print(f"\n⏱️ Total Execution Time: {execution_time:.5f} seconds")
print(f"\n🎯 Correct Initial Condition: h(0) = {s_correct:.6f}")

print("\n📊 Error Analysis Summary:")
print(f"Maximum Absolute Error: {max_abs_error:.2e}")
print(f"Maximum Relative Error: {max_rel_error:.4f}%")
print(f"Mean Absolute Error: {mean_abs_error:.2e}")
print(f"Mean Relative Error: {mean_rel_error:.4f}%")
print(f"RMS Error: {rms_error:.2e}")

# Print comparison table (rounded for readability)
pd.set_option("display.max_rows", None)
df_rounded = df.round(6)
print("\n📋 Detailed Comparison Table:")
print(df_rounded)

# ======================= PLOTS =======================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 1. RK4 Results Only
ax1.plot(xs, hs, color='blue', linewidth=2, label="h(x)")
ax1.scatter([x0, x1], [hs[0], 0], color='red', s=50, label="Boundary conditions", zorder=5)
ax1.set_title("Corneal Curvature Profile - RK4 Solution")
ax1.set_xlabel("x")
ax1.set_ylabel("h(x)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Comparison Graph
ax2.plot(xs, hs, color='blue', linewidth=2, label="RK4 Solution")
ax2.plot(ref_x, ref_h, 'o--', color='red', markersize=4, label="Reference Method")
ax2.set_title("RK4 vs Reference Method Comparison")
ax2.set_xlabel("x")
ax2.set_ylabel("h(x)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Additional detailed error analysis at specific points
print(f"\n🔍 Error Analysis at Key Points:")
key_indices = [0, len(xs)//4, len(xs)//2, 3*len(xs)//4, -1]
for i in key_indices:
    if i < len(xs):
        print(f"x = {xs[i]:.2f}: RK4 = {hs[i]:.6f}, Ref = {ref_h_interp[i]:.6f}, "
              f"Error = {abs_error[i]:.2e} ({rel_error[i]:.4f}%)")
