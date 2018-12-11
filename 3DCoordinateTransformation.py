# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 10:57:55 2018

@author: lelem
"""

import numpy as np
import matplotlib.pyplot as plt 
import math
# constant
XYZ = np.array([[1.0, 1.0, 1.0],\
                [4.0, 1.0, 0.5],\
                [1.0, 4.0, 2.0],\
                [4.0, 4.0, -1.0],\
                [3.0, 3.0, -0.5],\
                [3.5, 3.5, 1.0]])
# observation
xyz = np.array([[6.122, 0.372, 1.696],\
                [7.043, -2.252, 1.064],\
                [8.808, 1.485, 2.324],\
                [9.965, -1.330, -0.621],\
                [8.576, -0.510, -0.279],\
                [9.258, -0.984, 1.301]])

n = 18
n0 = 7
n_p = 6
# plot XY and xy to estimate the rotation angle Kappa for z
plt.plot(XYZ[:, 0], XYZ[:, 1], 'bo')
plt.plot(xyz[:, 0],xyz[:, 1], 'r+')
plt.xlabel("X")
plt.ylabel("Y")

fname = 'XY_xy_figure.png'
plt.savefig(fname)

plt.show()


# the initial estimate is 45 degree
Kappa0 = 45/180 * np.pi
Omega0 = 0
Phi0 = 0
Scale0 = 1 #
# initial estimate for tx, ty, tz:
R_Omega = np.array([[1, 0, 0],\
                    [0, np.cos(Omega0), np.sin(Omega0)],\
                    [0, -np.sin(Omega0), np.cos(Omega0)]])

R_Phi = np.array([[np.cos(Phi0), 0, -np.sin(Phi0)],\
                    [0, 1, 0],\
                    [np.sin(Phi0), 0, np.cos(Phi0)]])

R_Kappa = np.array([[np.cos(Kappa0), np.sin(Kappa0), 0],\
                    [-np.sin(Kappa0), np.cos(Kappa0), 0],\
                    [0, 0, 1]])

# use the first pair of point to estimate the translation
t0 = xyz[0, :].T - Scale0 * np.dot(np.dot(R_Kappa, R_Phi), np.dot(R_Omega, XYZ[0, :].T))  
print("Initial translation:")
print(t0)

## Least Squre
num = 0
while True:
    B = np.zeros((n,n0))
    f = np.zeros((n, 1))
    x = np.zeros((n0, 1))
    
    R_Omega = np.array([[1, 0, 0],\
                    [0, np.cos(Omega0), np.sin(Omega0)],\
                    [0, -np.sin(Omega0), np.cos(Omega0)]])

    R_Phi = np.array([[np.cos(Phi0), 0, -np.sin(Phi0)],\
                       [0, 1, 0],\
                       [np.sin(Phi0), 0, np.cos(Phi0)]])

    R_Kappa = np.array([[np.cos(Kappa0), np.sin(Kappa0), 0],\
                         [-np.sin(Kappa0), np.cos(Kappa0), 0],\
                         [0, 0, 1]])
    # B:
    for i in range(n_p):
        B1Column =  np.dot(np.dot(R_Kappa, R_Phi), np.dot(R_Omega, XYZ[i, :].T))
        # scale
        B11 = B1Column[0]
        B21 = B1Column[1]
        B31 = B1Column[2]
        # translation
        B15 = 1
        B26 = 1
        B37 = 1
        
        # rotation angle - Omega
        R_Omega_prime = np.array([[0, 0, 0],\
                    [0, -np.sin(Omega0), np.cos(Omega0)],\
                    [0, -np.cos(Omega0), -np.sin(Omega0)]])
        B2Column =  Scale0 * np.dot(np.dot(R_Kappa, R_Phi), np.dot(R_Omega_prime, XYZ[i, :].T))
        B12 = B2Column[0]
        B22 = B2Column[1]
        B32 = B2Column[2]
        
        # rotation angle - Phi
        R_Phi_prime = np.array([[-np.sin(Phi0), 0, -np.cos(Phi0)],\
                           [0, 0, 0],\
                           [np.cos(Phi0), 0, -np.sin(Phi0)]])
        B3Column =  Scale0 * np.dot(np.dot(R_Kappa, R_Phi_prime), np.dot(R_Omega, XYZ[i, :].T))
        B13 = B3Column[0]
        B23 = B3Column[1]
        B33 = B3Column[2]        
        
        # rotation angle - Kappa
        R_Kappa_prime = np.array([[-np.sin(Kappa0), np.cos(Kappa0), 0],\
                             [-np.cos(Kappa0), -np.sin(Kappa0), 0],\
                             [0, 0, 0]])
        B4Column =  Scale0 * np.dot(np.dot(R_Kappa_prime, R_Phi), np.dot(R_Omega, XYZ[i, :].T))
        B14 = B4Column[0]
        B24 = B4Column[1]
        B34 = B4Column[2]
        
        # f
        l = xyz[i, :].T - Scale0 * np.dot(np.dot(R_Kappa, R_Phi), np.dot(R_Omega, XYZ[i, :].T)) - t0 
        
        B[3 * i][0] = B11
        B[3 * i][1] = B12
        B[3 * i][2] = B13
        B[3 * i][3] = B14
        B[3 * i][4] = B15
        
        B[3 * i + 1][0] = B21
        B[3 * i + 1][1] = B22
        B[3 * i + 1][2] = B23
        B[3 * i + 1][3] = B24
        B[3 * i + 1][5] = B26        
        
        B[3 * i + 2][0] = B31
        B[3 * i + 2][1] = B32
        B[3 * i + 2][2] = B33
        B[3 * i + 2][3] = B34
        B[3 * i + 2][6] = B37 
        

        f[3 * i][0] = l[0]
        f[3 * i + 1][0] = l[1]
        f[3 * i + 2][0] = l[2]
        

    # correction for the parameters
    x = np.dot(np.linalg.inv(np.dot(B.T, B)), np.dot(B.T, f))
    num = num + 1
    # update
    Scale0 = Scale0 + x[0]
    Omega0 = Omega0 + x[1]
    Phi0 = Phi0 + x[2]
    Kappa0 = Kappa0 + x[3]
    t0[0] = t0[0] + x[4]
    t0[1] = t0[1] + x[5]
    t0[2] = t0[2] + x[6]
    
    v = np.dot(B, x) - f
    
    print("The {0} th interation:".format(num))
    print("The correction of parameters are: ")
    print(x)
    print("The correction of observations are: ")
    print(v)
    print("##############################")
    MinCorrection = min(abs(x))
    if MinCorrection < 0.0001:
        break


print("The final scale is:")
print(Scale0)
print("The rotation angle omega, phi, kappa are:")
print(Omega0)
print(Phi0)
print(Kappa0)
print("The translation are:")
print(t0)

