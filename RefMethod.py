import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters
R = 1.0
k = 1.0
P = 1.0
T = 1.0
a = R**2 * k / T
b = R * P / T
nx = 21
xl = 1.0
dx = xl / (nx - 1)
xg = np.linspace(0, xl, nx)
u0 = np.full(nx, 0.25)

# 6th-order finite difference for dh/dx
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

# 6th-order finite difference for d²h/dx²
def dss046(xl, xu, n, u, ux, nl, nu):
    dx = (xu - xl) / (n - 1)
    rdxs = 1 / dx**2
    uxx = np.zeros_like(u)

    if nl == 2:
        uxx[0] = rdxs * (-7.49388888888886*u[0]+12*u[1]-7.5*u[2]+4.44444444444457*u[3]-
                         1.875*u[4]+0.48*u[5]-0.055555555555568*u[6]-4.9*ux[0]*dx)
    uxx[1] = rdxs * (0.7*u[0]-0.388888888888889*u[1]-2.7*u[2]+4.75*u[3]-
                     3.72222222222222*u[4]+1.8*u[5]-0.5*u[6]+0.061111111111111*u[7])
    uxx[2] = rdxs * (-0.061111111111111*u[0]+1.18888888888889*u[1]-2.1*u[2]+
                     0.722222222222223*u[3]+0.472222222222222*u[4]-0.3*u[5]+
                     0.088888888888889*u[6]-0.011111111111111*u[7])
    for i in range(3, n-3):
        uxx[i] = rdxs * (0.011111111111111*u[i-3]-0.15*u[i-2]+1.5*u[i-1] +
                         1.5*u[i+1]-0.15*u[i+2]+0.011111111111111*u[i+3] -
                         2.72222222222222*u[i])
    uxx[n-3] = rdxs * (-0.061111111111111*u[n-1]+1.18888888888889*u[n-2]-2.1*u[n-3]+
                       0.722222222222223*u[n-4]+0.472222222222222*u[n-5]-0.3*u[n-6]+
                       0.088888888888889*u[n-7]-0.011111111111111*u[n-8])
    uxx[n-2] = rdxs * (0.7*u[n-1]-0.388888888888889*u[n-2]-2.7*u[n-3]+4.75*u[n-4]-
                       3.72222222222222*u[n-5]+1.8*u[n-6]-0.5*u[n-7]+0.061111111111111*u[n-8])
    if nu == 1:
        uxx[n-1] = rdxs * (5.21111111111111*u[n-1]-22.3*u[n-2]+43.95*u[n-3]-
                           52.7222222222222*u[n-4]+41*u[n-5]-20.1*u[n-6]+
                           5.66111111111111*u[n-7]-0.7*u[n-8])
    return uxx

# PDE system (ncase = 1)
def corneal_1(t, u):
    u[-1] = 0
    ux = dss006(0, xl, nx, u)
    ux[0] = 0
    uxx = dss046(0, xl, nx, u, ux, nl=2, nu=1)
    sr = np.sqrt(1 + ux**2)
    ut = uxx / sr - a * u + b / sr
    ut[-1] = 0
    return ut

# Time integration
tf = 1.0
nout = 6
tm = np.linspace(0, tf, nout)
sol = solve_ivp(corneal_1, [0, tf], u0, t_eval=tm, method='LSODA', rtol=1e-8, atol=1e-8)

# Output
U = sol.y.T
print(f"{'t':>5} {'u(0)':>10} {'u(0.25)':>10} {'u(0.5)':>10} {'u(0.75)':>10} {'u(1)':>10}")
for i in range(nout):
    print(f"{tm[i]:5.2f} {U[i, 0]:10.5f} {U[i, 5]:10.5f} {U[i, 10]:10.5f} {U[i, 15]:10.5f} {U[i, 20]:10.5f}")

# Plot
plt.figure(figsize=(8, 5))
for i in range(nout):
    plt.plot(xg, U[i], label=f't={tm[i]:.2f}')
plt.xlabel("x")
plt.ylabel("h(x,t)")
plt.title("Corneal Shape h(x,t) over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
