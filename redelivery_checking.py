
import numpy as np 
import pandas as pd 
import time
from docplex.mp.model import Model
from scipy.spatial import distance_matrix
df = pd.read_csv("/Users/Gym/Desktop/research/Benchmark_NTW_20/lr106.csv")

cust_size = df.shape[0] - 1
n = cust_size
g = [df["c_ready"][i] for i in range(n+1)]
h = [df["c_end"][i] for i in range(n+1)]
arr = [df["arr"][i] for i in range(n+1)]
nodeRedelivery = []
for i in range(1, n+1):
    if arr[i] < g[i]:
        nodeRedelivery.append(i)
    elif h[i] < arr[i]:
        nodeRedelivery.append(i)
Q = 10000
cust_size = df.shape[0]-1
n = nodeRedelivery
N = [i for i in n]
V = [0] + N 
K = [i for i in range(1,26)]
q = {i: 1 for i in N}
df2 = df.iloc[:, 1:3]
dist_m = pd.DataFrame(distance_matrix(df2.values, df2.values),index=df2.index, columns=df2.index)
time_start = time.time()

mdl = Model()

A = [(i, j) for i in V for j in V]
d = {(i, j): dist_m[i][j] for i, j in A}
t = {(i, j): dist_m[i][j] for i, j in A}
x = mdl.binary_var_dict(A, name = "x")
s = mdl.continuous_var_dict(N, name = "s")
T = 480
u = mdl.continuous_var_dict(N, ub= Q, name= 'u')
time = mdl.continuous_var_dict(V, ub = T, name= "time")
# print(time)

# Define objective function:
mdl.minimize(mdl.sum(((t[i,j]*0.18)+(d[i,j]*0.22))*x[i,j] for i, j in A)) 
# Add constraints:
mdl.add_constraints(mdl.sum(x[i,j] for j in V if j != i) == 1 for i in N) # Each point must be visited
mdl.add_constraints(mdl.sum(x[i,j] for i in V if i != j) == 1 for j in N) # Each point must be left
mdl.add_indicator_constraints(mdl.indicator_constraint(x[i,j], u[i]+q[j] == u[j]) for i,j in A if i!=0 and j!=0)
mdl.add_constraints(u[i] >= q[i] for i in N)
mdl.add_indicator_constraints(mdl.indicator_constraint(x[i,j], time[i]+t[i,j] + 5 == time[j]) for i,j in A if i != 0 and j != 0)
mdl.parameters.timelimit.set(10)

#Solving model:
solution = mdl.solve(log_output=True)
print(solution)
# print(solution.solve_status)

# active_arcs =[a for a in A if x[a].solution_value> 0.8]
# print(active_arcs)
print(len(n))
