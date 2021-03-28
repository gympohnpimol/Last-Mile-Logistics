
import os
import numpy as np 
import pandas as pd 
import sklearn as sk 
import matplotlib.pyplot as plt 
import time
from scipy.spatial import distance_matrix
import sys
from matplotlib import interactive
interactive(True)
from scipy.optimize import minimize
from docplex.mp.model import Model
from docplex.cp.model import CpoModel


# df = pd.read_excel("/Users/gympohnpimol/Desktop/research/Data/Excel_Dataset_S.xlsx")
df = pd.read_excel("/Users/gympohnpimol/Desktop/research/Data/Excel_Dataset_L.xlsx")

cust_size = df.shape[0] - 1
n = cust_size
e = [df["Earliest_Arr"][i] for i in range(n+1)]
l = [df["Latest_Arr"][i] for i in range(n+1)] # Latest time service
s = [df["ServiceTime"][i] for i in range(n+1)] # Service time
q = df["Demand"][0]
g = [df["c_ready"][i] for i in range(n+1)]
h = [df["c_end"][i] for i in range(n+1)]
arr = [df["arr"][i] for i in range(n+1)]

dist_m = pd.read_csv("/Users/gympohnpimol/Desktop/research/Data/distance_matrix.csv",encoding='latin1', error_bad_lines=False, header=None)
dist_m.loc[n+1,:] = dist_m.loc[0,:]
dist_m.loc[:,n+1] = dist_m.loc[:,0]

time_m = pd.read_csv("/Users/gympohnpimol/Desktop/research/Data/time_matrix.csv", encoding='latin1', error_bad_lines=False, header=None)
time_m.loc[n+1,:] = time_m.loc[0,:]
time_m.loc[:,n+1] = time_m.loc[:,0]
routeNode = [0, 4, 17, 16, 0]
# routeNode = [0, 6, 5, 3, 15, 12, 16, 17, 20, 13, 14, 0]
# routeNode = [0, 11, 7, 8, 18, 19, 9, 4, 1, 2, 10, 0]
# routeNode = [0, 10,3,13,20 ,0]
# print(arr[11], h[11])
service = 5
# def redelivery(e, l, g, h, arr, routeNode):
nodeRedelivery = []
for i in routeNode:
    if e[i] < arr[i] < g[i]:
        nodeRedelivery.append(i)
    elif h[i] < arr[i]:
        nodeRedelivery.append(i)
    elif l[i] <= arr[i]:
        nodeRedelivery.append(i)
    # elif arr[i] > e[i]:
    #     nodeRedelivery.append(i)
print(nodeRedelivery)

#arr = [0, 6.0, 15.0, 28.0, 40.0, 59.0, 82.0, 97.0, 107.0, 117.0, 124.0]
def check(e, l, g, h, time_m, routeNode, arr):
    global ArrivingTime, DepartureTime, node, nodeRedelivery
    def start(e, l, g, h, time_m):
        ArrivingTime = [0]
        DepartureTime = [0]
        node = [0]
        from_node = routeNode[0]
        to_node = routeNode[1]

        if e[to_node] <= time_m[from_node][to_node] < l[to_node]:
            arrivingTime = time_m[from_node][to_node]
            departTime = time_m[from_node][to_node] + service
        elif l[to_node] < time_m[from_node][to_node]:
            arrivingTime = time_m[from_node][to_node]
            departTime = time_m[from_node][to_node]
        ArrivingTime.append(arrivingTime)
        DepartureTime.append(departTime)
        node.append(to_node)
        return ArrivingTime, DepartureTime, node
    ArrivingTime, DepartureTime, node  = start(e, l, g, h, time_m)
    print(node)
    print(ArrivingTime)
    print(DepartureTime)


    def toNode(e, l, g, h,time_m, routeNode):
        from_node = node[-1]

        for i in range(1, len(routeNode)-1):
            ETD = DepartureTime.pop(-1)
            # print(routeNode[i], " -> ", routeNode[i+1])
            to_node = routeNode[i+1]
            if from_node != to_node and to_node not in node and e[to_node] <= ETD + time_m[from_node][to_node] < l[to_node]:
                arrivingTime = ETD + time_m[from_node][to_node]
                departTime = ETD + time_m[from_node][to_node] + service
            elif from_node != to_node and to_node not in node and l[to_node] < ETD +time_m[from_node][to_node]:
                arrivingTime = ETD + time_m[from_node][to_node]
                departTime = ETD + time_m[from_node][to_node]
            elif from_node != to_node and to_node not in node and e[to_node] > ETD + time_m[from_node][to_node]:
                arrivingTime = ETD + time_m[from_node][to_node]
                departTime = ETD + time_m[from_node][to_node]+ service
            ArrivingTime.append(arrivingTime)
            DepartureTime.append(departTime)
        return ArrivingTime, DepartureTime, node  

    ArrivingTime, DepartureTime, node= toNode(e, l, g, h, time_m, routeNode)
    # print(node)
    # print(ArrivingTime) 

    def redelivery(e, l, g, h, arr):
        nodeRedelivery = []
        for i in routeNode:
            if e[i] < arr[i] < g[i]:
                nodeRedelivery.append(i)
            elif h[i] <= arr[i] < l[i]:
                nodeRedelivery.append(i)
            elif l[i] < arr[i]:
                nodeRedelivery.append(i)
            # elif arr[i] > e[i]:
            #     nodeRedelivery.append(i)
        return nodeRedelivery
    nodeRedelivery = redelivery(e, l, g, h, arr)
    # print(nodeRedelivery)
check(e, l, g, h, time_m, routeNode, arr)    

R = [0, 11, 6, 5, 8, 3, 15, 12, 18, 4, 9]
R2 = [0, 4, 3, 17, 20, 15, 13, 14, 12, 16, 18, 19, 9, 0]

def routes(e, l, g, h, time_m):
    global selectedData, ArrivingTime, DepartureTime, node, nodeRedelivery
    def begin(e, g, h, time_m):
        selectedData = []
        ArrivingTime = [0]
        DepartureTime = []
        node = [0]
        for from_node in R:
            for to_node in R:
                if from_node == 0 and to_node != 0 and e[to_node] <= time_m[from_node][to_node] <= l[to_node]:
                    time = time_m[from_node][to_node]
                    selectedData.append((to_node, time)) 
        timeArrival = min(selectedData, key = lambda x:x[1])[1]
        nodeArrival = min(selectedData, key = lambda x:x[1])[0]  
        timeDepart = timeArrival + service
        ArrivingTime.append(timeArrival)
        node.append(nodeArrival)
        DepartureTime.append(timeDepart)
        return selectedData, ArrivingTime, DepartureTime, node
    
    selectedData, ArrivingTime, DepartureTime, node = begin(e, g, h, time_m)
    # print(node)

    def next(e, l, g, h, time_m, selectedData, ArrivingTime, DepartureTime, node):
        ETD = DepartureTime.pop(-1)
        selectedData.clear()
        # for from_node in node:
        from_node = node[-1] 
        for to_node in range(0,21):
            if from_node != to_node and to_node not in node and e[to_node] <= ETD + time_m[from_node][to_node]< l[to_node]:
                arrvingTime = ETD + time_m[from_node][to_node]
                departTime = ETD + time_m[from_node][to_node] + service
                selectedData.append((to_node, arrvingTime))
            elif from_node != to_node and to_node not in node and ETD + time_m[from_node][to_node] > l[to_node]:
                arrvingTime = ETD + time_m[from_node][to_node]
                departTime = ETD + time_m[from_node][to_node]
                selectedData.append((to_node, arrvingTime))
        timeArrival = min(selectedData, key = lambda x:x[1])[1]
        nodeArrival = min(selectedData, key = lambda x:x[1])[0]   
        ArrivingTime.append(timeArrival)
        node.append(nodeArrival)
        DepartureTime.append(timeArrival)
        return selectedData, ArrivingTime, DepartureTime, node
    for i in range(0,19):
        selectedData, ArrivingTime, DepartureTime, node = next(e, l, g, h, time_m, selectedData, ArrivingTime, DepartureTime, node)
    # print(selectedData)
    # print(node)


    # def redelivery(e, l, g, h, time_m, selectedData, ArrivingTime, DepartureTime, node):
    #     nodeRedelivery = []
    #     for i in routeNode:
    #         if g[i] > ArrivingTime[i] + service:
    #             nodeRedelivery.append(i)
    #         elif h[i] < ArrivingTime[i] < l[i]:
    #             nodeRedelivery.append(i)
    #     return nodeRedelivery
    # nodeRedelivery = redelivery(e, l, g, h, time_m, selectedData, ArrivingTime, DepartureTime, node)
    # print(nodeRedelivery)
        
routes(e, l, g, h, time_m)


#     for to_node in N:
#         if from_node == to_node-1 or from_node+10 == to_node-1: continue
#         arr = depart[from_node] + time_m[from_node][to_node-1]
#         arrival.append({"from_node": from_node+1, "to_node": to_node, "time": arr})

        
# arr = []
# a = depart[0] + time_m[0][1]
# a = [ arr.append(arrival[i:i+18]) for i in range(0, len(arrival), 18)]
# print(arr[0][0]["from_node"], "->", arr[0][0]["to_node"], "=", arr[0][0]["time"])
# print(arr[0][0]["from_node"])



