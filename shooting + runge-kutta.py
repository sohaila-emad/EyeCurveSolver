import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters
a = 1.0
b = 1.0
x0 = 0.0
x1 = 1.0
h_step = 0.05

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

# Shooting method to match boundary h(1) = 0
def shooting_method(s1, s2, tol=1e-6):
    _, _, _, y1 = integrate(s1)
    _, _, _, y2 = integrate(s2)
    while abs(s2 - s1) > tol:
        s3 = s2 - y2 * (s2 - s1) / (y2 - y1)
        _, _, _, y3 = integrate(s3)
        s1, y1 = s2, y2
        s2, y2 = s3, y3
    return s2

# Run shooting to find correct h(0)
s_correct = shooting_method(0.1, 0.5)

# Final integration with correct h(0)
xs, hs, hs_prime, _ = integrate(s_correct)

# Create a table and display it
df = pd.DataFrame({
    "x": xs,
    "h(x)": hs,
    "h'(x)": hs_prime
})

# Round and print table
pd.set_option("display.max_rows", None)  # Show all rows
df_rounded = df.round(5)
print("\n📊 Corneal Curvature Table:")
print(df_rounded)

# Optional: plot the curve
plt.plot(xs, hs)
plt.xlabel("x")
plt.ylabel("h(x)")
plt.title("Corneal Curvature Profile")
plt.grid(True)
plt.show()
