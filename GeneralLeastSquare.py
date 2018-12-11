# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 15:36:36 2018

@author: lelem
"""

# read data
import numpy as np

filepath = 'data_Sphere.txt'
observations = []
xc = 0
yc = 0
zc = 0
R = 0

with open(filepath) as fp:
    for line in fp:
        values = line.split()
        x = float(values[0])
        y = float(values[1])
        z = float(values[2])
        observations.append([x, y, z])
        
observations = np.asarray(observations, dtype=np.float32)

n_p = 156 
n = 156 * 3
c = n_p
n0 = 4 + 2 * n_p
r = n - n0
u = 4


# initial estimation
xyz_mean = np.mean(observations, axis = 0)
xc = xyz_mean[0]
yc = xyz_mean[1]
zc = xyz_mean[2]
R = np.sqrt((xc - x)**2 +(yc - y)**2 +(zc - z)**2 )

# Mixed Model
w = np.eye(n)
num = 0
while True:
    
    B = np.zeros([c, u])
    A = np.zeros([c, n])
    f = np.zeros([c, 1])
    
    for i in range(c):
        xo = observations[i][0]
        yo = observations[i][1]
        zo = observations[i][2]
        Ro = np.sqrt((xc - xo)**2 +(yc - yo)**2 +(zc - zo)**2 )
        
        A[i][i * 3] = (xo - xc) / Ro
        A[i][i * 3 + 1] = (yo - yc) / Ro
        A[i][i * 3 + 2] = (zo - zc) / Ro
        
        f[i] = -Ro + R
        
        B[i][0] = -(xo - xc) / Ro
        B[i][1] = -(yo - yc) / Ro
        B[i][2] = - (zo - zc) / Ro
        B[i][3] = -1
        
    Q = np.linalg.inv(w)
    Qe = np.dot(np.dot(A, Q), A.T)
    We = np.linalg.inv(Qe)
    
    delta = np.dot(np.linalg.inv(np.dot(np.dot(B.T,We), B)), np.dot(np.dot(B.T,We), f))
    K = np.dot(We, (f - np.dot(B, delta)))
    v = np.dot(np.dot(Q, A.T), K)
    
    
    # Correct the observation:
    for i in range(c):
        observations[i][0] = observations[i][0] + v[3 * (i - 1)]
        observations[i][1] = observations[i][1] + v[3 * (i - 1) + 1]
        observations[i][2] = observations[i][2] + v[3 * (i - 1) + 2]
         
    
    xc = xc + delta[0]
    yc = yc + delta[1]
    zc = zc + delta[2]
    R = R + delta[3]
    
    num = num + 1
    
    delta_abs = np.abs(delta)
    delta_abs_max = np.max(delta_abs)
    print(delta_abs_max)
    
    if num > 30 or delta_abs_max < 0.0001:
        break

v_abs = np.abs(v)
v_abs_max = np.max(v_abs)
print(v_abs_max)