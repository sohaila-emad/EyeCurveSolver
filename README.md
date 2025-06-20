# EyeCurveSolver

A comprehensive numerical analysis toolkit for solving corneal curvature differential equations using multiple computational methods including Physics-Informed Neural Networks (PINNs), Method of Lines (MOL), Newton-Raphson with Finite Difference Method (FDM), and Shooting Method with Runge-Kutta integration.

## Overview

EyeCurveSolver implements and compares various numerical methods for solving the nonlinear differential equation that governs corneal shape:

```
d²h/dx² / √(1 + (dh/dx)²) - a·h + b/√(1 + (dh/dx)²) = 0
```

This equation models the curvature of the corneal surface under specific boundary conditions, which is crucial for understanding eye biomechanics and vision correction applications.

## Features

- **Multiple Solution Methods**: Implementation of 4 different numerical approaches
- **Physics-Informed Neural Networks (PINN)**: Modern deep learning approach with automatic differentiation
- **Method of Lines (MOL)**: High-order finite difference with 6th-order spatial derivatives
- **Newton-Raphson + FDM**: Classical iterative method with finite difference discretization
- **Shooting + Runge-Kutta**: Boundary value problem solver with 4th-order RK integration
- **Comprehensive Comparison**: Detailed accuracy and performance analysis between methods
- **Visualization Tools**: Professional plots for solution comparison and error analysis
- **Performance Metrics**: Execution time, convergence analysis, and error statistics

## Methods Implemented

### 1. Physics-Informed Neural Network (PINN)
- **File**: `MLmethod.py`
- **Features**: Deep learning with physics constraints, automatic differentiation
- **Network**: 4-layer neural network with 50 neurons per layer
- **Training**: Adam optimizer with physics, data, and boundary condition losses

### 2. Method of Lines (MOL) - Reference Solution
- **File**: `RefMethod.py`
- **Features**: 6th-order finite differences for spatial derivatives
- **Integration**: LSODA solver for time evolution to steady state
- **Accuracy**: High-precision reference solution

### 3. Newton-Raphson + Finite Difference Method
- **File**: `Newton-Raphson + FDM.py`
- **Features**: Iterative nonlinear solver with numerical Jacobian
- **Convergence**: Damped Newton method with convergence monitoring
- **Boundary Conditions**: Mixed Neumann and Dirichlet conditions

### 4. Shooting Method + Runge-Kutta
- **File**: `shooting + runge-kutta.py`
- **Features**: Boundary value problem converted to initial value problem
- **Integration**: 4th-order Runge-Kutta with adaptive step size
- **Root Finding**: Secant method for boundary condition satisfaction

## Installation

### Prerequisites

```bash
# Core dependencies
pip install numpy scipy matplotlib torch pandas
```

### Setup

1. Clone the repository:
```bash
git clone https://github.com/sohaila-emad/EyeCurveSolver.git
cd EyeCurveSolver
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running Individual Methods

#### PINN Method
```bash
python MLmethod.py
```
- Trains neural network to learn the solution
- Compares with MOL reference solution
- Provides comprehensive accuracy metrics

#### Reference Method (MOL)
```bash
python RefMethod.py
```
- Generates high-accuracy reference solution
- Shows time evolution to steady state
- Uses 6th-order spatial discretization

#### Newton-Raphson + FDM
```bash
python "Newton-Raphson + FDM.py"
```
- Iterative solution with convergence tracking
- Displays residual reduction per iteration
- Provides error analysis at reference points

#### Shooting + Runge-Kutta
```bash
python "shooting + runge-kutta.py"
```
- Boundary value problem solution
- Comprehensive error analysis
- Statistical comparison with reference

### Example Output

Each method provides detailed results including:

```
Method Comparison Results:
- Execution Time: X.XXXX seconds
- Maximum Absolute Error: X.XXe-XX
- Maximum Relative Error: X.XXXX%
- Convergence Iterations: XX
```

## Mathematical Model

The solver addresses the steady-state corneal curvature equation:

**Differential Equation:**
```
d²h/dx² / √(1 + (dh/dx)²) - a·h + b/√(1 + (dh/dx)²) = 0
```

**Boundary Conditions:**
- `dh/dx|_{x=0} = 0` (Neumann condition at center)
- `h(1) = 0` (Dirichlet condition at edge)

**Parameters:**
- `a = R²k/T = 1.0` (curvature parameter)
- `b = RP/T = 1.0` (pressure parameter)
- Domain: `x ∈ [0, 1]`

## File Structure

```
EyeCurveSolver/
├── MLmethod.py                    # PINN implementation
├── RefMethod.py                   # Method of Lines (reference)
├── RefMethodApprox.py            # Simplified MOL version
├── Newton-Raphson + FDM.py       # Newton-Raphson solver
├── shooting + runge-kutta.py     # Shooting method
├── README.md                     # This file
└── requirements.txt              # Dependencies
```

## Performance Comparison

| Method | Typical Time | Accuracy | Parameters | Complexity |
|--------|-------------|----------|------------|------------|
| MOL | ~0.1s | Reference | 21-101 points | Medium |
| PINN | ~20-400s | High | ~5000 params | High |
| Newton-Raphson | ~0.01s | High | 21 points | Low |
| Shooting+RK4 | ~0.01s | High | Adaptive | Low |

## Key Results

The methods typically achieve:
- **Maximum Absolute Error**: < 1e-4
- **Maximum Relative Error**: < 0.1%
- **Convergence**: 10-50 iterations for iterative methods
- **PINN Training**: 2000 epochs for optimal accuracy

## Visualization

Each method generates plots showing:
- Solution profiles h(x)
- Comparison with reference solution
- Error analysis and convergence behavior
- Training progress (for PINN)

## Applications

This solver is relevant for:
- **Ophthalmology**: Corneal shape analysis and modeling
- **Vision Correction**: Contact lens and surgical planning
- **Biomechanics**: Eye tissue mechanical properties
- **Numerical Methods Research**: Comparative analysis of solution techniques

## Contributing

Contributions are welcome! Areas for improvement:
- Additional numerical methods
- Adaptive mesh refinement
- Parameter sensitivity analysis
- 2D/3D extensions

## Dependencies

- `numpy`: Numerical computations
- `scipy`: Scientific computing and integration
- `matplotlib`: Plotting and visualization
- `torch`: Deep learning framework for PINN
- `pandas`: Data analysis and tabulation

## Citation

If you use this code in research, please cite:

```bibtex
@software{eyecurvesolver2024,
  author = {Sohaila Emad},
  title = {EyeCurveSolver: Multi-Method Numerical Analysis for Corneal Curvature Equations},
  url = {https://github.com/sohaila-emad/EyeCurveSolver},
  year = {2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Sohaila Emad** - [sohaila-emad](https://github.com/sohaila-emad)

## Acknowledgments

- Physics-Informed Neural Networks methodology
- High-order finite difference schemes
- Classical numerical analysis techniques
- Scientific computing community

---

**Note**: This project demonstrates the application of both classical and modern numerical methods to biomedical engineering problems, specifically corneal biomechanics modeling.
