import numpy as np
import matplotlib.pyplot as plt
import time

# coefficients
a = 1.0
b = 1.0
xl = 1.0
N = 21
dx = xl / (N - 1)
x = np.linspace(0, xl, N)

# boundary conditions
h0 = 0.25
hN = 0.0

# initial approximation
h = np.linspace(h0, hN, N)

# reference solution to compare with
x_ref = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
h_ref = np.array([0.34803, 0.32771, 0.26570, 0.15864, 0.0])

def residual(h):
    res = np.zeros_like(h)
    res[0] = (h[1] - h[0]) / dx  # Neumann BC
    for i in range(1, N - 1):
        dhdx = (h[i + 1] - h[i - 1]) / (2 * dx)
        d2hdx2 = (h[i + 1] - 2 * h[i] + h[i - 1]) / dx ** 2
        denom = np.sqrt(1 + dhdx ** 2)
        res[i] = -d2hdx2 + a * h[i] - b / denom
    res[-1] = h[-1] - hN  # Dirichlet BC
    return res

# Initialize variables to store max errors
max_abs_error_history = []
max_rel_error_history = []

start_time = time.time()

max_iterations = 150
iteration_count = 0
alpha = 0.5  # ✅ Damping factor

for iteration in range(max_iterations):
    iteration_count += 1
    F = residual(h)

    # Calculate errors at reference points
    idx_ref = [np.argmin(np.abs(x - xi)) for xi in x_ref]
    h_num = h[idx_ref]
    abs_error = np.abs(h_num - h_ref)

    rel_error = np.zeros_like(h_ref)
    mask = h_ref != 0
    rel_error[mask] = np.abs((h_num[mask] - h_ref[mask]) / h_ref[mask])

    current_max_abs = np.max(abs_error)
    current_max_rel = np.max(rel_error[mask]) if np.any(mask) else 0.0
    max_abs_error_history.append(current_max_abs)
    max_rel_error_history.append(current_max_rel)

    print(f"\nIteration {iteration_count} (||F||_inf = {np.linalg.norm(F, np.inf):.2e}):")
    print("x\t\th(x)\t\tAbs Error\tRel Error")
    for xi, hi, aerr, rerr in zip(x_ref, h_num, abs_error, rel_error):
        rel_error_str = f"{rerr:.6f}" if h_ref[np.where(x_ref == xi)[0][0]] != 0 else "N/A"
        print(f"{xi:.2f}\t{hi:.6f}\t{aerr:.6f}\t{rel_error_str}")

    if np.linalg.norm(F, np.inf) < 1e-8:
        print("\nSolution converged!")
        break

    # Compute Jacobian numerically
    J = np.zeros((N, N))
    delta = 1e-5  # ✅ More stable for finite differences
    for j in range(N):
        h_pert = h.copy()
        h_pert[j] += delta
        J[:, j] = (residual(h_pert) - F) / delta

    dh = np.linalg.solve(J, -F)
    h += alpha * dh  # ✅ Damped update

end_time = time.time()
execution_time = end_time - start_time

print("\nFinal Results:")
print("x\tNumerical\tReference\tAbs Error\tRel Error")
for xi, hi, href in zip(x_ref, h_num, h_ref):
    aerr = np.abs(hi - href)
    rerr = np.abs((hi - href) / href) if href != 0 else np.nan
    rel_error_str = f"{rerr:.6f}" if not np.isnan(rerr) else "N/A"
    print(f"{xi:.2f}\t{hi:.6f}\t{href:.6f}\t{aerr:.6f}\t{rel_error_str}")

final_max_abs = max_abs_error_history[-1] if max_abs_error_history else 0.0
final_max_rel = max_rel_error_history[-1] if max_rel_error_history else 0.0

print("\nFinal Analysis:")
print(f"Maximum Absolute Error: {final_max_abs:.6f}")
print(f"Maximum Relative Error: {final_max_rel:.6f}")
print(f"\nNumber of iterations: {iteration_count}")
print(f"Execution time: {execution_time:.6f} seconds")

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, h, 'b-', linewidth=2, label='Numerical Solution')
plt.plot(x_ref, h_ref, 'ro', markersize=8, label='Reference Points')
plt.xlabel('x', fontsize=12)
plt.ylabel('h(x)', fontsize=12)
plt.title('Steady-State Solution with Error Tracking', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.semilogy(range(1, iteration_count + 1), max_abs_error_history, 'bo-', label='Max Absolute Error')
if any(max_rel_error_history):
    plt.semilogy(range(1, iteration_count + 1), max_rel_error_history, 'r*-', label='Max Relative Error')
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Error', fontsize=12)
plt.title('Error Convergence', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()
