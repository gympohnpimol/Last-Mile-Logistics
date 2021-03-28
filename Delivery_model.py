import sys
import os
import numpy as np
import pandas as pd
import time
from docplex.mp.model import Model
from scipy.spatial import distance_matrix


df = pd.read_csv("/Users/gym/Desktop/research/Benchmark_NTW/lr106.csv")

Q = 10000
cust_size = df.shape[0]-1
n = cust_size
N = [i for i in range(1, n+1)]
V = [0] + N 
K = [i for i in range(1,26)]
q = {i: 1 for i in N}
df2 = df.iloc[:, 1:3]
dist_m = pd.DataFrame(distance_matrix(df2.values, df2.values),index=df2.index, columns=df2.index)
time_start = time.time()

e = [df["READY"][i] for i in range(n+1)]
e.append(df["READY"][0])
l = [df["DUE"][i] for i in range(n+1)]
l.append(df["DUE"][0])
ser = [df["SERVICE"][i] for i in range(n+1)]
ser.append(df["SERVICE"][0])

mdl = Model()

A = [(i, j) for i in V for j in V]
d = {(i, j): dist_m[i][j] for i, j in A}
t = {(i, j): dist_m[i][j] for i, j in A}
x = mdl.binary_var_dict(A, name = "x")
s = mdl.continuous_var_dict(N, name = "s")
T = 480
u = mdl.continuous_var_dict(N, ub= Q, name= 'u')
time = mdl.continuous_var_dict(V, ub = T, name= "time")


# Define objective function:
mdl.minimize(mdl.sum(((t[i,j]*0.18)+(d[i,j]*0.22))*x[i,j] for i, j in A)) 
# mdl.minimize(mdl.sum((d[i,j])*x[i,j] for i, j in A)) 
# mdl.minimize(mdl.sum((t[i,j])*x[i,j] for i, j in A)) 
# Add constraints:
mdl.add_constraints(mdl.sum(x[i,j] for j in V if j != i) == 1 for i in N) # Each point must be visited
mdl.add_constraints(mdl.sum(x[i,j] for i in V if i != j) == 1 for j in N) # Each point must be left
mdl.add_indicator_constraints(mdl.indicator_constraint(x[i,j], u[i]+q[j] == u[j]) for i,j in A if i!=0 and j!=0)
mdl.add_constraints(u[i] >= q[i] for i in N)
mdl.add_indicator_constraints(mdl.indicator_constraint(x[i,j], time[i]+t[i,j] + ser[i] == time[j]) for i,j in A if i != 0 and j != 0)
mdl.add_constraints(time[j] >= e[j]  for j in N if j != 0)
mdl.add_constraints(time[j] <= l[j]  for j in N if j != 0)
mdl.parameters.timelimit.set(10)

# Solving model:
solution = mdl.solve(log_output=True)
print(solution)

active_arcs = [a for a in A if x[a].solution_value > 0.9]
print(active_arcs)



