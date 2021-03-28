
import scipy.spatial
import libpysal as ps
import numpy as np
import pandas as pd 
from pointpats import PointPattern, PoissonClusterPointProcess, as_window
import math
df = pd.read_csv("/Users/Gym/Desktop/research/Benchmark_LTW_20/lrc106.csv")
x = list(df['XCOORD.'])
y = list(df['YCOORD.'])
point = (zip(x,y))
# print(list(point))

pp = PointPattern(point)
print(pp.summary())
# print(pp.mean_nnd)

a = math.sqrt(7790/106)
print(pp.mean_nnd/(0.5*a))