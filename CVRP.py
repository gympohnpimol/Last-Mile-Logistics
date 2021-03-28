
import numpy as np
from docplex.mp.model import Model
import matplotlib.pyplot as plt

import sys
import os

# Define a problem:
rnd = np.random
rnd.seed(0)

n = 10
Q = 15
N = [i for i in range(1, n+1)]
V = [0] + N
q = {i:rnd.randint(1,10) for i in N}

loc_x = rnd.rand(len(V))*200
loc_y = rnd.rand(len(V))*100

A = [(i, j) for i in V for j in V if i!=j] # List of Arcs
c = {(i,j): round(np.hypot(loc_x[i]-loc_x[j], loc_y[i]-loc_y[j])) for i, j in A} # Dictionary of distances/costs

# Create a CPLEX model:
mdl = Model('CVRP')

# Define arcs and capacities:
x = mdl.binary_var_dict(A, name= 'x')
u = mdl.continuous_var_dict(N, ub= Q, name= 'u')

# Define objective function:
mdl.minimize(mdl.sumsq(c[i,j]*x[i,j] for i, j in A))

# Add constraints:
mdl.add_constraints(mdl.sum(x[i,j] for j in V if j != i) == 1 for i in N) # Each point must be visited
mdl.add_constraints(mdl.sum(x[i,j] for i in V if i != j) == 1 for j in N) # Each point must be left
mdl.add_indicator_constraints(mdl.indicator_constraint(x[i,j], u[i]+q[j] == u[j]) for i,j in A if i!=0 and j!=0)
mdl.add_constraints(u[i] >= q[i] for i in N)
mdl.parameters.timelimit = 100 # Add running time limit

# Solving model:
solution = mdl.solve(log_output=True)

print(solution)
print(solution.solve_status) # Returns if the solution is Optimal or just Feasible

active_arcs = [a for a in A if x[a].solution_value > 0.9]
print(active_arcs)

# Plot solution:
# plt.scatter(loc_x[1:], loc_y[1:], c='b')
# for i in N:
#     plt.annotate('$q_%d=%d$'%(i,q[i]), (loc_x[i]+2,loc_y[i]))
# for i, j in active_arcs:
#     plt.plot([loc_x[i], loc_x[j]], [loc_y[i], loc_y[j]], c='g', alpha=0.3)
# plt.plot(loc_x[0], loc_y[0], c='r', marker='s')
# plt.axis('equal')
# plt.show()

# export PYTHONPATH=/Applications/CPLEX_Studio_Community129/cplex/python/3.7/x86-64_osx