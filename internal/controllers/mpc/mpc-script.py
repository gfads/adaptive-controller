import cvxpy as cp
import numpy as np
# import matplotlib.pyplot as plt

# System parameters (simple linear model)
a = 0.9  # effect of previous load
b = 0.1  # effect of control-theory input

# MPC parameters
N = 10  # prediction horizon
T = 50  # total simulation time
x_target = 0.7  # target CPU utilization

# Constraints
u_min, u_max = 1.0, 3.0  # GHz range
x_min, x_max = 0.0, 1.0  # CPU load bounds

# Weights for the cost function
Q = 1.0   # penalty on deviation from target
R = 0.01  # penalty on control-theory effort

# Initialization
x = 0.5  # initial CPU load
x_history = [x]
u_history = []

for t in range(T):
    # Define variables for optimization
    x_var = cp.Variable(N + 1)
    u_var = cp.Variable(N)

    # Define constraints and objective
    constraints = [x_var[0] == x]
    cost = 0
    for k in range(N):
        # Dynamics constraint
        constraints += [x_var[k + 1] == a * x_var[k] + b * u_var[k]]
        # Bounds
        constraints += [u_min <= u_var[k], u_var[k] <= u_max]
        constraints += [x_min <= x_var[k + 1], x_var[k + 1] <= x_max]
        # Cost function
        cost += Q * cp.square(x_var[k + 1] - x_target) + R * cp.square(u_var[k])

    # Solve optimization problem
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()

    # Apply the first control-theory input
    u = u_var.value[0]
    x = a * x + b * u

    # Record history
    x_history.append(x)
    u_history.append(u)


print(x_history[0])
