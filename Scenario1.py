#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp
import math
import matplotlib.pyplot as plt


# In[2]:


##################################
# horizon
I1 = 245
N = 20
I = I1-N
ds = 1

# lane radius 
LR1 = 360/math.pi
LR2 = 360/math.pi + 3.6

Rs1 = 5
Rs2 = 186
Rs3 = 136 # merging lane centerline length
Rs4 = 146 # merging lane leftline length
Rs5 = 126 # merging lane rightline length

# speed limit
p_des = np.ones(I1+1)/20

# alphaa limit
alpha_des = np.zeros(I1+1)

##################################
# main lane length
Ls1 = np.zeros(I1+1)
for i in range (I1):
    Ls1[i+1] = Ls1[i] + 1

# lane left and right limits 
Ll = np.ones(I1+1)*(-1.8)
Lr = np.ones(I1+1)*5.4
for i in range (135,155):
    Lr[i+1] = Lr[i] - 0.18
for i in range (156,I1+1):
    Lr[i] = 1.8

# main lane centerline X-axis
Lx1 = np.zeros(I1+1)
for i in range (Rs1):
    Lx1[i+1] = Lx1[i] + 1
for i in range (Rs1+1, Rs2):
    Lx1[i] = Rs1 + LR1*math.sin((i-Rs1)/LR1)
for i in range (Rs2, I1+1):
    Lx1[i] = Rs1 + LR1

# main lane centerline Y-axis
Ly1 = np.zeros(I1+1)
for i in range (Rs1+1):
    Ly1[i] = 5.4
for i in range (Rs1+1, Rs2):
    Ly1[i] = LR1 + 5.4 - LR1*math.cos((i-Rs1)/LR1)
for i in range (Rs2, I1+1):
    Ly1[i] = Ly1[i-1] + 1

# main theta_des
theta1_des = np.zeros(I1+1)
for i in range (Rs1+1, Rs2):
    theta1_des[i] = (i-Rs1)/LR1
for i in range (Rs2, I1+1):
    theta1_des[i] = math.pi/2

Lc1 = np.zeros(I1+1)
for i in range (Rs1+1, I1+1):
    Lc1[i] = -math.cos(math.pi/2 + theta1_des[i])
Ld1 = np.zeros(I1+1)
for i in range (Rs2):
    Ld1[i] = math.sin(math.pi/2 + theta1_des[i])

# main lane leftline X&Y-axis
Lxl1 = Lx1 - 1.8 * Lc1
Lyl1 = Ly1 + 1.8 * Ld1

# main lane rightline X&Y-axis
Lxr1 = Lx1 + 1.8 * Lc1
Lyr1 = Ly1 - 1.8 * Ld1

# main k_des
k1_des = np.zeros(I1+1)
for i in range (Rs1+1, Rs2):
    k1_des[i] = 1/LR1

##################################
# merging lane length
Ls2 = np.zeros(Rs3)
for i in range (Rs3-1):
    Ls2[i+1] = Ls2[i] + 1

# merging lane centerline X-axis
Lx2 = np.zeros(Rs3)
for i in range (Rs1):
    Lx2[i+1] = Lx2[i] + 1
for i in range (Rs1+1, Rs3):
    Lx2[i] = Rs1 + LR2*math.sin((i-Rs1)/LR2)

# merging lane centerline Y-axis
Ly2 = np.ones(Rs3)*1.8
for i in range (Rs1+1, Rs3):
    Ly2[i] = LR2 + 1.8 - LR2*math.cos((i-Rs1)/LR2)

# merging theta_des
theta2_des = np.zeros(Rs4)
for i in range (Rs1+1, Rs4):
    theta2_des[i] = (i-Rs1)/LR2

Lc2 = np.zeros(Rs4)
for i in range (Rs1+1, Rs4):
    Lc2[i] = -math.cos(math.pi/2 + theta2_des[i])
Ld2 = np.zeros(Rs4)
for i in range (Rs4):
    Ld2[i] = math.sin(math.pi/2 + theta2_des[i])

# merging lane rightline X-axis
Lxr2 = np.zeros(Rs4)
for i in range (Rs5):
    Lxr2[i] = Lx2[i] + 1.8 * Lc2[i]
for i in range (Rs5, Rs4):
    Lxr2[i] = Lxr1[i] + (3.42-0.18*(i-Rs5)) * Lc2[i]

# merging lane rightline Y-axis
Lyr2 = np.zeros(Rs4)
for i in range (Rs5):
    Lyr2[i] = Ly2[i] - 1.8 * Ld2[i]
for i in range (Rs5, Rs4):
    Lyr2[i] = Lyr1[i] - (3.42-0.18*(i-Rs5)) * Ld2[i]

# merging lane k_des
k2_des = np.zeros(I1+1)
for i in range (Rs1+1, I1+1):
    k2_des[i] = 1/LR2
##################################


# In[3]:


##################################
# Create variables
J = 4
r_max = 3.6
tau_star = 1 
tau_min = 0.4
ls = 2
phi_min = -math.pi/6
phi_max = math.pi/6
a_min = -5
a_max = 3
R_min = 10
eps = 0.0001
M = 20

# result
Y = np.zeros((J,I1+1,4))
U = np.zeros((J,I1,2))

# vehicle dynamics
time = np.zeros((J,I1+1))
pace = np.zeros((J,I1+1))
speed = np.zeros((J,I1+1))
alpha = np.zeros((J,I1))
acceleration = np.zeros((J,I1))
angular = np.zeros((J,I1))
k = np.zeros((J,I1))

#initial
# for n in range(7):
#     Y[1,n,2] = 3.6
#     Y[3,n,2] = 3.8

Y[0,0,1] = -1/180

Y[1,2,0] = -0.1
Y[1,2,1] = 0
Y[1,2,2] = 3.6

Y[2,4,0] = 0.1
Y[2,4,1] = 0

Y[3,6,0] = 0.3
Y[3,6,1] = -0.005
Y[3,6,2] = 3.6

pace[0,0] = p_des[0] - Y[0,0,1]
speed[0,0] = 1/pace[0,0]

################################
# diagonal matrices
W = sp.diags([1., 10., 0., 0.1])
Q = sp.diags([10000., 1.])
S = sp.diags([5., 50., 0., 0.5])

W1 = sp.diags([1., 10., 0.0001, 0.1])
Q1 = sp.diags([10000., 1.])
S1 = sp.diags([5., 50., 0.03, 0.5])

W0 = sp.diags([10., 0.0001, 0.1])
Q0 = sp.diags([10000., 1.])
S0 = sp.diags([50., 0.03, 0.5])

# Discretization for leading vehicle
A = sp.csc_matrix([
[1., 0., 0.],
[0., 1., 1.],
[0., 0., 1.]])

B = sp.csc_matrix([
[1., 0.],
[0., 1/2],
[0., 1.]])

# Discretization
Ad = sp.csc_matrix([
[1., 1., 0., 0.],
[0., 1., 0., 0.],
[0., 0., 1., 1.],
[0., 0., 0., 1.]])

Bd = sp.csc_matrix([
[1/2, 0.],
[1., 0.],
[0., 1/2],
[0., 1.]])
################################


# In[4]:


################################
# vehicle 1 Loop
for i in range(0,7):
    for j in range(0,1):
        #print(j)
        MODEL = gp.Model("LaneChange")
        
        ################################ gurobi variables
        y = MODEL.addMVar(shape=(N+1,3),lb=-GRB.INFINITY,name='y')
        u = MODEL.addMVar(shape=(N,2),lb=-GRB.INFINITY,name='u')
        MODEL.update()
        
        ################################ gurobi objective
        MODEL.addConstr(y[0,0] == -Y[j,i,1])
        MODEL.addConstr(y[0,1] == Y[j,i,2])
        MODEL.addConstr(y[0,2] == Y[j,i,3])
        for n in range(N):
            MODEL.addConstr(y[n+1, :] == A @ y[n, :] + B @ u[n, :])
        for n in range(N):
            ################################ gurobi constraints
            # x
            MODEL.addConstr(1/180 <= y[n+1,0])
            MODEL.addConstr(y[n+1,0] <= 1)
            # x
            MODEL.addConstr(Ll[i+n+1] <= y[n+1,1])
            MODEL.addConstr(y[n+1,1] <= Lr[i+n+1])
            # x
            MODEL.addConstr(phi_min <= y[n+1,2])
            MODEL.addConstr(y[n+1,2] <= phi_max)
            # u
            MODEL.addConstr(-a_max*pace[j,i]**3 <= u[n,0])
            MODEL.addConstr(u[n,0] <= -a_min*pace[j,i]**3)
            # u
            MODEL.addConstr(-1/R_min - k1_des[i+n] <= u[n,1])
            MODEL.addConstr(u[n,1] <= 1/R_min - k1_des[i+n])
            ################################
        
        obj1 = y[N, :] @ S0 @ y[N, :] 
        obj2 = sum(y[n, :] @ W0 @ y[n, :] for n in range(N))
        obj3 = sum(u[n, :] @ Q0 @ u[n, :] for n in range(N))
        MODEL.setObjective(obj1 + obj2 + obj3, GRB.MINIMIZE)
        MODEL.optimize()
        
        for n in range(N):
            Y[j,i+n+1,1] = -y[n+1,0].X
            Y[j,i+n+1,2] = y[n+1,1].X
            Y[j,i+n+1,3] = y[n+1,2].X
            U[j,i+n,:] = u[n,:].X 
            
            pace[j,i+n+1] = p_des[i+n+1] + y[n+1,0].X
            speed[j,i+n+1] = 1/pace[j,i+n+1]
            
            alpha[j,i+n] = alpha_des[i+n] + u[n,0].X
            acceleration[j,i+n] = (speed[j,i+n+1]**2 - speed[j,i+n]**2)/2
            
            k[j,i+n] = u[n,1].X + k1_des[n]
            time[j,i+n+1] = time[j,i+n] + 2/(speed[j,i+n] + speed[j,i+n+1])
################################


# In[5]:


################################
time[1,ls] = time[0,0] + tau_star - Y[1,ls,0]
pace[1,ls] = pace[0,0] - Y[1,ls,1]
speed[1,ls] = 1/pace[1,ls]
# vehicle 2 Loop
for i in range(2,7):
    for j in range(1,2):
        #print(j)
        MODEL = gp.Model("LaneChange")
        
        ################################ gurobi variables
        y = MODEL.addMVar(shape=(N+1,4),lb=-GRB.INFINITY,name='y')
        u = MODEL.addMVar(shape=(N,2),lb=-GRB.INFINITY,name='u')
        MODEL.update()
        
        ################################ gurobi objective
        MODEL.addConstr(y[0,:] == Y[j,i,:])
        for n in range(N):
            MODEL.addConstr(y[n+1, :] == Ad @ y[n, :] + Bd @ u[n, :])
        for n in range(N):
            ################################ gurobi constraints
            #10a
            MODEL.addConstr(y[n+1,0] <= tau_min)
            #10c
            MODEL.addConstr(y[n+1,1] <= pace[j-1,i+n+1-ls] - p_des[i+n+1-ls])
            #10d
            MODEL.addConstr(3.6 <= y[n+1,2])
            MODEL.addConstr(y[n+1,2] <= Lr[i+n+1])
            #10e
            MODEL.addConstr(phi_min <= y[n+1,3])
            MODEL.addConstr(y[n+1,3] <= phi_max)
            #10f
            MODEL.addConstr(alpha[j-1,i+n-ls] + a_min*pace[j,i]**3 <= u[n,0])
            MODEL.addConstr(u[n,0] <= alpha[j-1,i+n-ls] + a_max*pace[j,i]**3)
            #10g
            MODEL.addConstr(-1/R_min - k2_des[i+n] <= u[n,1])
            MODEL.addConstr(u[n,1] <= 1/R_min - k2_des[i+n])
            ################################
        
        obj1 = y[N, :] @ S @ y[N, :] 
        obj2 = sum(y[n, :] @ W @ y[n, :] for n in range(N))
        obj3 = sum(u[n, :] @ Q @ u[n, :] for n in range(N))
        MODEL.setObjective(obj1 + obj2 + obj3, GRB.MINIMIZE)
        MODEL.optimize()
        
        for n in range(N):
            Y[j,i+n+1,:] = y[n+1,:].X
            U[j,i+n,:] = u[n,:].X 
            
            time[j,i+n+1] = time[j-1,i+n+1-ls] + tau_star - y[n+1,0].X
            
            pace[j,i+n+1] = pace[j-1,i+n+1-ls] - y[n+1,1].X
            speed[j,i+n+1] = 1/pace[j,i+n+1]
            
            alpha[j,i+n] = alpha[j-1,i+n-ls] - u[n,0].X
            acceleration[j,i+n] = (speed[j,i+n+1]**2 - speed[j,i+n]**2)/2
            
            k[j,i+n] = u[n,1].X + k2_des[n]
################################


# In[6]:


################################
time[2,2*ls] = time[1,ls] + tau_star - Y[2,2*ls,0]
pace[2,2*ls] = pace[1,ls] - Y[2,2*ls,1]
speed[2,2*ls] = 1/pace[2,2*ls]
# vehicle 3 Loop
for i in range(4,7):
    for j in range(2,3):
        #print(j)
        MODEL = gp.Model("LaneChange")
        
        ################################ gurobi variables
        y = MODEL.addMVar(shape=(N+1,4),lb=-GRB.INFINITY,name='y')
        u = MODEL.addMVar(shape=(N,2),lb=-GRB.INFINITY,name='u')
        MODEL.update()
        
        ################################ gurobi objective
        MODEL.addConstr(y[0,:] == Y[j,i,:])
        for n in range(N):
            MODEL.addConstr(y[n+1, :] == Ad @ y[n, :] + Bd @ u[n, :])
        for n in range(N):
            ################################ gurobi constraints
            #10a
            MODEL.addConstr(y[n+1,0] <= tau_min)
            #10c
            MODEL.addConstr(y[n+1,1] <= pace[j-1,i+n+1-ls] - p_des[i+n+1-ls])
            #10d
            MODEL.addConstr(Ll[i+n+1] <= y[n+1,2])
            MODEL.addConstr(y[n+1,2] <= Lr[i+n+1])
            #10e
            MODEL.addConstr(phi_min <= y[n+1,3])
            MODEL.addConstr(y[n+1,3] <= phi_max)
            #10f
            MODEL.addConstr(alpha[j-1,i+n-ls] + a_min*pace[j,i]**3 <= u[n,0])
            MODEL.addConstr(u[n,0] <= alpha[j-1,i+n-ls] + a_max*pace[j,i]**3)
            #10g
            MODEL.addConstr(-1/R_min - k1_des[i+n] <= u[n,1])
            MODEL.addConstr(u[n,1] <= 1/R_min - k1_des[i+n])
            ################################
        
        obj1 = y[N, :] @ S1 @ y[N, :] 
        obj2 = sum(y[n, :] @ W1 @ y[n, :] for n in range(N))
        obj3 = sum(u[n, :] @ Q1 @ u[n, :] for n in range(N))
        MODEL.setObjective(obj1 + obj2 + obj3, GRB.MINIMIZE)
        MODEL.optimize()
        
        for n in range(N):
            Y[j,i+n+1,:] = y[n+1,:].X
            U[j,i+n,:] = u[n,:].X 
            
            time[j,i+n+1] = time[j-1,i+n+1-ls] + tau_star - y[n+1,0].X
            
            pace[j,i+n+1] = pace[j-1,i+n+1-ls] - y[n+1,1].X
            speed[j,i+n+1] = 1/pace[j,i+n+1]
            
            alpha[j,i+n] = alpha[j-1,i+n-ls] - u[n,0].X
            acceleration[j,i+n] = (speed[j,i+n+1]**2 - speed[j,i+n]**2)/2
            
            k[j,i+n] = u[n,1].X + k1_des[n]
################################


# In[7]:


################################
time[3,3*ls] = time[2,2*ls] + tau_star - Y[3,3*ls,0]
pace[3,3*ls] = pace[2,2*ls] - Y[3,3*ls,1]
speed[3,3*ls] = 1/pace[3,3*ls]
# Loop
for i in range(6,I-2):
    print(i)
    for j in range(0,J):
        if j == 0:
            MODEL = gp.Model("LaneChange")

            ################################ gurobi variables
            y = MODEL.addMVar(shape=(N+1,3),lb=-GRB.INFINITY,name='y')
            u = MODEL.addMVar(shape=(N,2),lb=-GRB.INFINITY,name='u')
            z = MODEL.addMVar(shape=(N,1),vtype=GRB.BINARY,name='z')
            b1 = MODEL.addMVar(shape=(N,1),vtype=GRB.BINARY,name='b1')
            b2 = MODEL.addMVar(shape=(N,1),vtype=GRB.BINARY,name='b2')
            b = MODEL.addMVar(shape=(N,1),vtype=GRB.BINARY,name='b')
            MODEL.update()

            ################################ gurobi objective
            MODEL.addConstr(y[0,0] == -Y[j,i,1])
            MODEL.addConstr(y[0,1] == Y[j,i,2])
            MODEL.addConstr(y[0,2] == Y[j,i,3])
            for n in range(N):
                MODEL.addConstr(y[n+1, :] == A @ y[n, :] + B @ u[n, :])
            for n in range(N):
                ################################ gurobi constraints
                # x
                MODEL.addConstr(1/180 <= y[n+1,0])
                MODEL.addConstr(y[n+1,0] <= 1)
                # x
                MODEL.addConstr(r_max/2 - Y[j+1,i+n+1+ls,2] >= (b1[n] - 1)*M)
                MODEL.addConstr(r_max/2 - Y[j+1,i+n+1+ls,2] <= b1[n]*M)
                MODEL.addConstr(Y[j+1,i+n+1+ls,0] - tau_min >= (b2[n] - 1)*M)
                MODEL.addConstr(Y[j+1,i+n+1+ls,0] - tau_min <= b2[n]*M)
                MODEL.addGenConstrAnd(b[n], [b1[n], b2[n]])
                MODEL.addConstr((b[n] == 0) >> (y[n+1,1] >= Ll[i+n+1]))
                MODEL.addConstr((b[n] == 1) >> (y[n+1,1] >= r_max))
                MODEL.addConstr(y[n+1,1] <= Lr[i+n+1])
                # x
                MODEL.addConstr(phi_min <= y[n+1,2])
                MODEL.addConstr(y[n+1,2] <= phi_max)
                # u
                MODEL.addConstr(-a_max*pace[j,i]**3 <= u[n,0])
                MODEL.addConstr(u[n,0] <= -a_min*pace[j,i]**3)
                # u
                MODEL.addConstr(y[n+1,1] - r_max/2 >= (z[n] - 1)*M)
                MODEL.addConstr(y[n+1,1] - r_max/2 <= z[n]*M)
                MODEL.addConstr((z[n] == 0) >> (-1/R_min - k1_des[i+n] <= u[n,1]))
                MODEL.addConstr((z[n] == 0) >> (u[n,1] <= 1/R_min - k1_des[i+n]))
                MODEL.addConstr((z[n] == 1) >> (-1/R_min - k2_des[i+n] <= u[n,1]))
                MODEL.addConstr((z[n] == 1) >> (u[n,1] <= 1/R_min - k2_des[i+n]))
                MODEL.addConstr(-1/R_min - k1_des[i+n] <= u[n,1])
                ################################

            obj1 = y[N, :] @ S0 @ y[N, :] 
            obj2 = sum(y[n, :] @ W0 @ y[n, :] for n in range(N))
            obj3 = sum(u[n, :] @ Q0 @ u[n, :] for n in range(N))
            MODEL.setObjective(obj1 + obj2 + obj3, GRB.MINIMIZE)
            MODEL.optimize()

            for n in range(N):
                Y[j,i+n+1,1] = -y[n+1,0].X
                Y[j,i+n+1,2] = y[n+1,1].X
                Y[j,i+n+1,3] = y[n+1,2].X
                U[j,i+n,:] = u[n,:].X 

                pace[j,i+n+1] = p_des[i+n+1] + y[n+1,0].X
                speed[j,i+n+1] = 1/pace[j,i+n+1]

                alpha[j,i+n] = alpha_des[i+n] + u[n,0].X
                acceleration[j,i+n] = (speed[j,i+n+1]**2 - speed[j,i+n]**2)/2

                time[j,i+n+1] = time[j,i+n] + 2/(speed[j,i+n] + speed[j,i+n+1])
                if z[n].X == 0:
                    k[j,i+n] = u[n,1].X + k1_des[n]
                else:
                    k[j,i+n] = u[n,1].X + k2_des[n]
        else:
            
            W2 = sp.diags([1., 10., math.exp(0.1*(i-Rs3)), 0.1])
            Q2 = sp.diags([10000., 1.])
            S2 = sp.diags([5., 50., 5*math.exp(0.1*(i-Rs3)), 0.5])
            
            MODEL = gp.Model("LaneChange")
            ################################ gurobi variables
            y = MODEL.addMVar(shape=(N+1,4),lb=-GRB.INFINITY,name='y')
            u = MODEL.addMVar(shape=(N,2),lb=-GRB.INFINITY,name='u')
            z1 = MODEL.addMVar(shape=(N,1),vtype=GRB.BINARY,name='z1')
            z2 = MODEL.addMVar(shape=(N,1),vtype=GRB.BINARY,name='z2')
            b1 = MODEL.addMVar(shape=(N,1),vtype=GRB.BINARY,name='b1')
            b2 = MODEL.addMVar(shape=(N,1),vtype=GRB.BINARY,name='b2')
            b3 = MODEL.addMVar(shape=(N,1),vtype=GRB.BINARY,name='b3')
            if j < J-1:
                b4 = MODEL.addMVar(shape=(N,1),vtype=GRB.BINARY,name='b4')
                b5 = MODEL.addMVar(shape=(N,1),vtype=GRB.BINARY,name='b5')
                b6 = MODEL.addMVar(shape=(N,1),vtype=GRB.BINARY,name='b6')
                b = MODEL.addMVar(shape=(N,1),vtype=GRB.BINARY,name='b')
            MODEL.update()

            ################################ gurobi objective
            MODEL.addConstr(y[0,:] == Y[j,i,:])
            for n in range(N):
                MODEL.addConstr(y[n+1, :] == Ad @ y[n, :] + Bd @ u[n, :])
            for n in range(N):
                ################################ gurobi constraints
                #10a
                MODEL.addConstr((Y[j-1,i+n+1-ls,2] - r_max/2)*(y[n+1,2] - r_max/2) >= (z1[n] - 1)*M)
                MODEL.addConstr((Y[j-1,i+n+1-ls,2] - r_max/2)*(y[n+1,2] - r_max/2) <= z1[n]*M)
                MODEL.addConstr((z1[n] == 0) >> (y[n+1,0] <= tau_star))
                MODEL.addConstr((z1[n] == 1) >> (y[n+1,0] <= tau_min))
                #10b
                MODEL.addConstr(y[n+1,1] <= pace[j-1,i+n+1-ls] - p_des[i+n+1-ls])
                #10c
                MODEL.addConstr(r_max/2 - Y[j-1,i+n+1-ls,2] >= (b1[n] - 1)*M)
                MODEL.addConstr(r_max/2 - Y[j-1,i+n+1-ls,2] <= b1[n]*M)
                MODEL.addConstr(y[n+1,0] - tau_min >= (b2[n] - 1)*M)
                MODEL.addConstr(y[n+1,0] - tau_min <= b2[n]*M)
                MODEL.addGenConstrAnd(b3[n], [b1[n], b2[n]])
                if j < J-1:
                    MODEL.addConstr(r_max/2 - Y[j+1,i+n+1+ls,2] >= (b4[n] - 1)*M)
                    MODEL.addConstr(r_max/2 - Y[j+1,i+n+1+ls,2] <= b4[n]*M)
                    MODEL.addConstr(Y[j+1,i+n+1+ls,0] - tau_min >= (b5[n] - 1)*M)
                    MODEL.addConstr(Y[j+1,i+n+1+ls,0] - tau_min <= b5[n]*M)
                    MODEL.addGenConstrAnd(b6[n], [b4[n], b5[n]])
                    MODEL.addGenConstrOr(b[n], [b3[n], b6[n]])
                    MODEL.addConstr((b[n] == 0) >> (y[n+1,2] >= Ll[i+n+1]))
                    MODEL.addConstr((b[n] == 1) >> (y[n+1,2] >= r_max))
                else:
                    MODEL.addConstr((b3[n] == 0) >> (y[n+1,2] >= Ll[i+n+1]))
                    MODEL.addConstr((b3[n] == 1) >> (y[n+1,2] >= r_max))
                MODEL.addConstr(y[n+1,2] <= Lr[i+n+1])
                #10d
                MODEL.addConstr(phi_min <= y[n+1,3])
                MODEL.addConstr(y[n+1,3] <= phi_max)
                #10e
                MODEL.addConstr(alpha[j-1,i+n-ls] + a_min*pace[j,i]**3 <= u[n,0])
                MODEL.addConstr(u[n,0] <= alpha[j-1,i+n-ls] + a_max*pace[j,i]**3)
                #10f
                MODEL.addConstr(y[n+1,2] - r_max/2 >= (z2[n] - 1)*M)
                MODEL.addConstr(y[n+1,2] - r_max/2 <= z2[n]*M)
                MODEL.addConstr((z2[n] == 0) >> (-1/R_min - k1_des[i+n] <= u[n,1]))
                MODEL.addConstr((z2[n] == 0) >> (u[n,1] <= 1/R_min - k1_des[i+n]))
                MODEL.addConstr((z2[n] == 1) >> (-1/R_min - k2_des[i+n] <= u[n,1]))
                MODEL.addConstr((z2[n] == 1) >> (u[n,1] <= 1/R_min - k2_des[i+n]))
                ################################
            if Y[j,i,2] < r_max/2:
                obj1 = y[N, :] @ S1 @ y[N, :] 
                obj2 = sum(y[n, :] @ W1 @ y[n, :] for n in range(N))
                obj3 = sum(u[n, :] @ Q1 @ u[n, :] for n in range(N))
            elif i < 36:
                obj1 = y[N, :] @ S @ y[N, :] 
                obj2 = sum(y[n, :] @ W @ y[n, :] for n in range(N))
                obj3 = sum(u[n, :] @ Q @ u[n, :] for n in range(N))
            else:
                obj1 = y[N, :] @ S2 @ y[N, :] 
                obj2 = sum(y[n, :] @ W2 @ y[n, :] for n in range(N))
                obj3 = sum(u[n, :] @ Q2 @ u[n, :] for n in range(N))
            MODEL.setObjective(obj1 + obj2 + obj3, GRB.MINIMIZE)
            MODEL.optimize()

            for n in range(N):
                Y[j,i+n+1,:] = y[n+1,:].X
                U[j,i+n,:] = u[n,:].X 

                time[j,i+n+1] = time[j-1,i+n+1-ls] + tau_star - y[n+1,0].X

                pace[j,i+n+1] = pace[j-1,i+n+1-ls] - y[n+1,1].X
                speed[j,i+n+1] = 1/pace[j,i+n+1]

                alpha[j,i+n] = alpha[j-1,i+n-ls] - u[n,0].X
                acceleration[j,i+n] = (speed[j,i+n+1]**2 - speed[j,i+n]**2)/2
                
                if z2[n].X == 0:
                    k[j,i+n] = u[n,1].X + k1_des[n]
                else:
                    k[j,i+n] = u[n,1].X + k2_des[n]
            for n in range (i+N+1, I1+1):
                Y[j,n,0] = Y[j,i+N,0]
                Y[j,n,2] = Y[j,i+N,2]
################################


# In[8]:


Rs0 = np.zeros(180)
for i in range (179):
    Rs0[i+1] = Rs0[i] + 1
time0 = np.zeros(180)
for i in range (180):
    time0[i] = time[0,i]

Rs1 = np.zeros(178)
Rs1[0] = 2
for i in range (177):
    Rs1[i+1] = Rs1[i] + 1
time1 = np.zeros(178)
for i in range (178):
    time1[i] = time[1,i+2]

Rs2 = np.zeros(176)
Rs2[0] = 4
for i in range (175):
    Rs2[i+1] = Rs2[i] + 1
time2 = np.zeros(176)
for i in range (176):
    time2[i] = time[2,i+4]

Rs3 = np.zeros(174)
Rs3[0] = 6
for i in range (173):
    Rs3[i+1] = Rs3[i] + 1
time3 = np.zeros(174)
for i in range (174):
    time3[i] = time[3,i+6]

lane1 = np.zeros((3,2,180))
lane21 = np.zeros((2,136))
lane22 = np.zeros((2,146))

space0 = np.zeros(180)
for i in range (179):
    space0[i+1] = space0[i] + 1
space1 = np.zeros(178)
for i in range (178):
    space1[i] = space0[i+2]
space2 = np.zeros(176)
for i in range (176):
    space2[i] = space1[i+2]
space3 = np.zeros(174)
for i in range (174):
    space3[i] = space2[i+2]

Vx0 = np.zeros(180)
Vx1 = np.zeros(178)
Vx2 = np.zeros(176)
Vx3 = np.zeros(174)
Vy0 = np.zeros(180)
Vy1 = np.zeros(178)
Vy2 = np.zeros(176)
Vy3 = np.zeros(174)

headway1 = np.zeros(178)
headway2 = np.zeros(176)
headway3 = np.zeros(174)
pacedeviation0 = np.zeros(180)
pacedeviation1 = np.zeros(178)
pacedeviation2 = np.zeros(176)
pacedeviation3 = np.zeros(174)
lateraldeviation0 = np.zeros(180)
lateraldeviation1 = np.zeros(178)
lateraldeviation2 = np.zeros(176)
lateraldeviation3 = np.zeros(174)
angulardeviation0 = np.zeros(180)
angulardeviation1 = np.zeros(178)
angulardeviation2 = np.zeros(176)
angulardeviation3 = np.zeros(174)
Speed0 = np.zeros(180)
Speed1 = np.zeros(178)
Speed2 = np.zeros(176)
Speed3 = np.zeros(174)
acceleration0 = np.zeros(180)
acceleration1 = np.zeros(178)
acceleration2 = np.zeros(176)
acceleration3 = np.zeros(174)
k0 = np.zeros(180)
k1 = np.zeros(178)
k2 = np.zeros(176)
k3 = np.zeros(174)

for i in range(136):
    lane21[0,i] = Lx2[i]
    lane21[1,i] = Ly2[i]

for i in range(146):
    lane22[0,i] = Lxr2[i]
    lane22[1,i] = Lyr2[i]

for i in range(180):
    lane1[0,0,i] = Lxl1[i]
    lane1[0,1,i] = Lyl1[i]
    lane1[1,0,i] = Lx1[i]
    lane1[1,1,i] = Ly1[i]
    lane1[2,0,i] = Lxr1[i]
    lane1[2,1,i] = Lyr1[i]

for i in range(180):
    Vx0[i] = lane1[1,0,i] + Y[0,i,2] * Lc1[i]
    Vy0[i] = lane1[1,1,i] - Y[0,i,2] * Ld1[i]
    pacedeviation0[i] = Y[0,i,1]
    lateraldeviation0[i] = Y[0,i,2]
    angulardeviation0[i] = Y[0,i,3]
    Speed0[i] = speed[0,i]
    acceleration0[i] = acceleration[0,i]
    k0[i] = k[0,i]

for i in range(178):
    headway1[i] = Y[1,i+2,0]
    Vx1[i] = lane1[1,0,i+2] + Y[1,i+2,2] * Lc1[i+2]
    Vy1[i] = lane1[1,1,i+2] - Y[1,i+2,2] * Ld1[i+2]
    pacedeviation1[i] = Y[1,i+2,1]
    lateraldeviation1[i] = Y[1,i+2,2]
    angulardeviation1[i] = Y[1,i+2,3]
    Speed1[i] = speed[1,i+2]
    acceleration1[i] = acceleration[1,i]
    k1[i] = k[1,i]

for i in range(176):
    headway2[i] = Y[2,i+4,0]
    Vx2[i] = lane1[1,0,i+4] + Y[2,i+4,2] * Lc1[i+4]
    Vy2[i] = lane1[1,1,i+4] - Y[2,i+4,2] * Ld1[i+4]
    pacedeviation2[i] = Y[2,i+4,1]
    lateraldeviation2[i] = Y[2,i+4,2]
    angulardeviation2[i] = Y[2,i+4,3]
    Speed2[i] = speed[2,i+4]
    acceleration2[i] = acceleration[2,i]
    k2[i] = k[2,i]

for i in range(174):
    headway3[i] = Y[3,i+6,0]
    Vx3[i] = lane1[1,0,i+6] + Y[3,i+6,2] * Lc1[i+6]
    Vy3[i] = lane1[1,1,i+6] - Y[3,i+6,2] * Ld1[i+6]
    pacedeviation3[i] = Y[3,i+6,1]
    lateraldeviation3[i] = Y[3,i+6,2]
    angulardeviation3[i] = Y[3,i+6,3]
    Speed3[i] = speed[3,i+6]
    acceleration3[i] = acceleration[3,i]
    k3[i] = k[3,i]

lane11 = np.zeros((2,30))
for i in range(30):
    lane11[0,i] = lane1[1,0,i]
    lane11[1,i] = lane1[1,1,i]
lane12 = np.zeros((2,116))
for i in range(116):
    lane12[0,i] = lane1[1,0,i+30]
    lane12[1,i] = lane1[1,1,i+30]
lane13 = np.zeros((2,34))
for i in range(34):
    lane13[0,i] = lane1[1,0,i+146]
    lane13[1,i] = lane1[1,1,i+146]

lane211 = np.zeros((2,30))
for i in range(30):
    lane211[0,i] = lane21[0,i]
    lane211[1,i] = lane21[1,i]
lane212 = np.zeros((2,106))
for i in range(106):
    lane212[0,i] = lane21[0,i+30]
    lane212[1,i] = lane21[1,i+30]

plt.plot(lane1[0,0,:], lane1[0,1,:], color='black')
plt.plot(lane11[0,:], lane11[1,:], color='black', linestyle='dashed', label='No lane-changing section')
plt.plot(lane12[0,:], lane12[1,:], color='magenta', linestyle='dashed', label='Lane-changing section')
plt.plot(lane13[0,:], lane13[1,:], color='black', linestyle='dashed')
plt.plot(lane1[2,0,:], lane1[2,1,:], color='black')
plt.plot(lane211[0,:], lane211[1,:], color='black', linestyle='dashed')
plt.plot(lane212[0,:], lane212[1,:], color='magenta', linestyle='dashed')
plt.plot(lane22[0,:], lane22[1,:], color='black')
plt.legend(fontsize="16", loc ="upper left")
plt.xlabel('Globe x (m)', fontsize="18")
plt.ylabel('Globe y (m)', fontsize="18")
plt.ylim(-5, 123)
plt.savefig('-1.png',bbox_inches="tight")
plt.show()


# In[9]:


plt.plot(time1, headway1, color='red', label='Vehicle 2')
plt.plot(time2, headway2, color='orange', label='Vehicle 3')
plt.plot(time3, headway3, color='green', label='Vehicle 4')
plt.legend(fontsize="16")
plt.ylim(-0.11, 0.31)
plt.rc('font', size=18)
plt.xlabel('Time (s)', fontsize="18")
plt.ylabel('Headway deviation (s)', fontsize="18")
plt.savefig('Headway.png',bbox_inches="tight")
plt.show()

plt.plot(time0, Speed0, color='blue', label='Vehicle 1')
plt.plot(time1, Speed1, color='red', label='Vehicle 2')
plt.plot(time2, Speed2, color='orange', label='Vehicle 3')
plt.plot(time3, Speed3, color='green', label='Vehicle 4')
plt.legend(fontsize="16")
plt.ylim(13.8, 20.2)
plt.rc('font', size=18)
plt.xlabel('Time (s)', fontsize="18")
plt.ylabel('Speed (m/s)', fontsize="18")
plt.savefig('Speed.png',bbox_inches="tight")
plt.show()

plt.plot(time0, lateraldeviation0, color='blue', label='Vehicle 1')
plt.plot(time1, lateraldeviation1, color='red', label='Vehicle 2')
plt.plot(time2, lateraldeviation2, color='orange', label='Vehicle 3')
plt.plot(time3, lateraldeviation3, color='green', label='Vehicle 4')
plt.legend(fontsize="16")
plt.ylim(-0.2, 4.2)
plt.rc('font', size=18)
plt.xlabel('Time (s)', fontsize="18")
plt.ylabel('Lateral deviation (m)', fontsize="18")
plt.savefig('Lateral Deviation.png',bbox_inches="tight")
plt.show()

plt.plot(time0, angulardeviation0, color='blue', label='Vehicle 1')
plt.plot(time1, angulardeviation1, color='red', label='Vehicle 2')
plt.plot(time2, angulardeviation2, color='orange', label='Vehicle 3')
plt.plot(time3, angulardeviation3, color='green', label='Vehicle 4')
plt.legend(fontsize="16", loc ="lower right")
plt.ylim(-0.13, 0.01)
plt.rc('font', size=18)
plt.xlabel('Time (s)', fontsize="18")
plt.ylabel('Angular deviation (rad)', fontsize="18")
plt.savefig('Angular Deviation.png',bbox_inches="tight")
plt.show()


# In[13]:


plt.plot(lane1[0,0,:], lane1[0,1,:], color='black')
plt.plot(lane11[0,:], lane11[1,:], color='black', linestyle='dashed')
plt.plot(lane12[0,:], lane12[1,:], color='magenta', linestyle='dashed')
plt.plot(lane13[0,:], lane13[1,:], color='black', linestyle='dashed')
plt.plot(lane1[2,0,:], lane1[2,1,:], color='black')
plt.plot(lane211[0,:], lane211[1,:], color='black', linestyle='dashed')
plt.plot(lane212[0,:], lane212[1,:], color='magenta', linestyle='dashed')
plt.plot(lane22[0,:], lane22[1,:], color='black')
plt.plot(Vx0[0], Vy0[0], 'bo', color='blue', label='Vehicle 1')
plt.legend(fontsize="16")
plt.xlabel('Globe x (m)', fontsize="18")
plt.ylabel('Globe y (m)', fontsize="18")
plt.title("$\it{t}$ = 0 s")
plt.savefig('0.png',bbox_inches="tight")
plt.show()

plt.plot(lane1[0,0,:], lane1[0,1,:], color='black')
plt.plot(lane11[0,:], lane11[1,:], color='black', linestyle='dashed')
plt.plot(lane12[0,:], lane12[1,:], color='magenta', linestyle='dashed')
plt.plot(lane13[0,:], lane13[1,:], color='black', linestyle='dashed')
plt.plot(lane1[2,0,:], lane1[2,1,:], color='black')
plt.plot(lane211[0,:], lane211[1,:], color='black', linestyle='dashed')
plt.plot(lane212[0,:], lane212[1,:], color='magenta', linestyle='dashed')
plt.plot(lane22[0,:], lane22[1,:], color='black')
plt.plot(Vx0[18], Vy0[18], 'bo', color='blue', label='Vehicle 1')
plt.legend(fontsize="16")
plt.xlabel('Globe x (m)', fontsize="18")
plt.ylabel('Globe y (m)', fontsize="18")
plt.title("$\it{t}$ = 1 s")
plt.savefig('1.png',bbox_inches="tight")
plt.show()

plt.plot(lane1[0,0,:], lane1[0,1,:], color='black')
plt.plot(lane11[0,:], lane11[1,:], color='black', linestyle='dashed')
plt.plot(lane12[0,:], lane12[1,:], color='magenta', linestyle='dashed')
plt.plot(lane13[0,:], lane13[1,:], color='black', linestyle='dashed')
plt.plot(lane1[2,0,:], lane1[2,1,:], color='black')
plt.plot(lane211[0,:], lane211[1,:], color='black', linestyle='dashed')
plt.plot(lane212[0,:], lane212[1,:], color='magenta', linestyle='dashed')
plt.plot(lane22[0,:], lane22[1,:], color='black')
plt.plot(Vx0[36], Vy0[36], 'bo', color='blue', label='Vehicle 1')
plt.plot(Vx1[17], Vy1[17], 'bo', color='red', label='Vehicle 2')
plt.plot(Vx2[0], Vy2[0], 'bo', color='orange', label='Vehicle 3')
plt.legend(fontsize="16")
plt.xlabel('Globe x (m)', fontsize="18")
plt.ylabel('Globe y (m)', fontsize="18")
plt.title("$\it{t}$ = 2 s")
plt.savefig('2.png',bbox_inches="tight")
plt.show()

plt.plot(lane1[0,0,:], lane1[0,1,:], color='black')
plt.plot(lane11[0,:], lane11[1,:], color='black', linestyle='dashed')
plt.plot(lane12[0,:], lane12[1,:], color='magenta', linestyle='dashed')
plt.plot(lane13[0,:], lane13[1,:], color='black', linestyle='dashed')
plt.plot(lane1[2,0,:], lane1[2,1,:], color='black')
plt.plot(lane211[0,:], lane211[1,:], color='black', linestyle='dashed')
plt.plot(lane212[0,:], lane212[1,:], color='magenta', linestyle='dashed')
plt.plot(lane22[0,:], lane22[1,:], color='black')
plt.plot(Vx0[54], Vy0[54], 'bo', color='blue', label='Vehicle 1')
plt.plot(Vx1[36], Vy1[36], 'bo', color='red', label='Vehicle 2')
plt.plot(Vx2[18], Vy2[18], 'bo', color='orange', label='Vehicle 3')
plt.plot(Vx3[5], Vy3[5], 'bo', color='green', label='Vehicle 4')
plt.legend(fontsize="16")
plt.xlabel('Globe x (m)', fontsize="18")
plt.ylabel('Globe y (m)', fontsize="18")
plt.title("$\it{t}$ = 3 s")
plt.savefig('3.png',bbox_inches="tight")
plt.show()

plt.plot(lane1[0,0,:], lane1[0,1,:], color='black')
plt.plot(lane11[0,:], lane11[1,:], color='black', linestyle='dashed')
plt.plot(lane12[0,:], lane12[1,:], color='magenta', linestyle='dashed')
plt.plot(lane13[0,:], lane13[1,:], color='black', linestyle='dashed')
plt.plot(lane1[2,0,:], lane1[2,1,:], color='black')
plt.plot(lane211[0,:], lane211[1,:], color='black', linestyle='dashed')
plt.plot(lane212[0,:], lane212[1,:], color='magenta', linestyle='dashed')
plt.plot(lane22[0,:], lane22[1,:], color='black')
plt.plot(Vx0[72], Vy0[72], 'bo', color='blue', label='Vehicle 1')
plt.plot(Vx1[54], Vy1[54], 'bo', color='red', label='Vehicle 2')
plt.plot(Vx2[36], Vy2[36], 'bo', color='orange', label='Vehicle 3')
plt.plot(Vx3[19], Vy3[19], 'bo', color='green', label='Vehicle 4')
plt.legend(fontsize="16")
plt.xlabel('Globe x (m)', fontsize="18")
plt.ylabel('Globe y (m)', fontsize="18")
plt.title("$\it{t}$ = 4 s")
plt.savefig('4.png',bbox_inches="tight")
plt.show()

plt.plot(lane1[0,0,:], lane1[0,1,:], color='black')
plt.plot(lane11[0,:], lane11[1,:], color='black', linestyle='dashed')
plt.plot(lane12[0,:], lane12[1,:], color='magenta', linestyle='dashed')
plt.plot(lane13[0,:], lane13[1,:], color='black', linestyle='dashed')
plt.plot(lane1[2,0,:], lane1[2,1,:], color='black')
plt.plot(lane211[0,:], lane211[1,:], color='black', linestyle='dashed')
plt.plot(lane212[0,:], lane212[1,:], color='magenta', linestyle='dashed')
plt.plot(lane22[0,:], lane22[1,:], color='black')
plt.plot(Vx0[90], Vy0[90], 'bo', color='blue', label='Vehicle 1')
plt.plot(Vx1[72], Vy1[72], 'bo', color='red', label='Vehicle 2')
plt.plot(Vx2[54], Vy2[54], 'bo', color='orange', label='Vehicle 3')
plt.plot(Vx3[36], Vy3[36], 'bo', color='green', label='Vehicle 4')
plt.legend(fontsize="16")
plt.xlabel('Globe x (m)', fontsize="18")
plt.ylabel('Globe y (m)', fontsize="18")
plt.title("$\it{t}$ = 5 s")
plt.savefig('5.png',bbox_inches="tight")
plt.show()

plt.plot(lane1[0,0,:], lane1[0,1,:], color='black')
plt.plot(lane11[0,:], lane11[1,:], color='black', linestyle='dashed')
plt.plot(lane12[0,:], lane12[1,:], color='magenta', linestyle='dashed')
plt.plot(lane13[0,:], lane13[1,:], color='black', linestyle='dashed')
plt.plot(lane1[2,0,:], lane1[2,1,:], color='black')
plt.plot(lane211[0,:], lane211[1,:], color='black', linestyle='dashed')
plt.plot(lane212[0,:], lane212[1,:], color='magenta', linestyle='dashed')
plt.plot(lane22[0,:], lane22[1,:], color='black')
plt.plot(Vx0[108], Vy0[108], 'bo', color='blue', label='Vehicle 1')
plt.plot(Vx1[90], Vy1[90], 'bo', color='red', label='Vehicle 2')
plt.plot(Vx2[72], Vy2[72], 'bo', color='orange', label='Vehicle 3')
plt.plot(Vx3[54], Vy3[54], 'bo', color='green', label='Vehicle 4')
plt.legend(fontsize="16")
plt.xlabel('Globe x (m)', fontsize="18")
plt.ylabel('Globe y (m)', fontsize="18")
plt.title("$\it{t}$ = 6 s")
plt.savefig('6.png',bbox_inches="tight")
plt.show()

plt.plot(lane1[0,0,:], lane1[0,1,:], color='black')
plt.plot(lane11[0,:], lane11[1,:], color='black', linestyle='dashed')
plt.plot(lane12[0,:], lane12[1,:], color='magenta', linestyle='dashed')
plt.plot(lane13[0,:], lane13[1,:], color='black', linestyle='dashed')
plt.plot(lane1[2,0,:], lane1[2,1,:], color='black')
plt.plot(lane211[0,:], lane211[1,:], color='black', linestyle='dashed')
plt.plot(lane212[0,:], lane212[1,:], color='magenta', linestyle='dashed')
plt.plot(lane22[0,:], lane22[1,:], color='black')
plt.plot(Vx0[126], Vy0[126], 'bo', color='blue', label='Vehicle 1')
plt.plot(Vx1[108], Vy1[108], 'bo', color='red', label='Vehicle 2')
plt.plot(Vx2[90], Vy2[90], 'bo', color='orange', label='Vehicle 3')
plt.plot(Vx3[72], Vy3[72], 'bo', color='green', label='Vehicle 4')
plt.legend(fontsize="16")
plt.xlabel('Globe x (m)', fontsize="18")
plt.ylabel('Globe y (m)', fontsize="18")
plt.title("$\it{t}$ = 7 s")
plt.savefig('7.png',bbox_inches="tight")
plt.show()

plt.plot(lane1[0,0,:], lane1[0,1,:], color='black')
plt.plot(lane11[0,:], lane11[1,:], color='black', linestyle='dashed')
plt.plot(lane12[0,:], lane12[1,:], color='magenta', linestyle='dashed')
plt.plot(lane13[0,:], lane13[1,:], color='black', linestyle='dashed')
plt.plot(lane1[2,0,:], lane1[2,1,:], color='black')
plt.plot(lane211[0,:], lane211[1,:], color='black', linestyle='dashed')
plt.plot(lane212[0,:], lane212[1,:], color='magenta', linestyle='dashed')
plt.plot(lane22[0,:], lane22[1,:], color='black')
plt.plot(Vx0[144], Vy0[144], 'bo', color='blue', label='Vehicle 1')
plt.plot(Vx1[126], Vy1[126], 'bo', color='red', label='Vehicle 2')
plt.plot(Vx2[108], Vy2[108], 'bo', color='orange', label='Vehicle 3')
plt.plot(Vx3[90], Vy3[90], 'bo', color='green', label='Vehicle 4')
plt.legend(fontsize="16")
plt.xlabel('Globe x (m)', fontsize="18")
plt.ylabel('Globe y (m)', fontsize="18")
plt.title("$\it{t}$ = 8 s")
plt.savefig('8.png',bbox_inches="tight")
plt.show()


# In[ ]:




