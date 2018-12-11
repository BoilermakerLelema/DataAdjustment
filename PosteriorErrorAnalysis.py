# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 11:23:29 2018

@author: lelem
"""
# read data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches

def plot_cov_ellipse(cov, pos, ax, Cp2_chi2, color, nstd = 2):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.
    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    
    vals, vecs = eigsorted(cov)
    
    print('eigenvalues:', vals)
    
    a = np.sqrt(vals[0] * Cp2_chi2)
    b = np.sqrt(vals[1] * Cp2_chi2)
    
    #theta_new = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))


    # Width and height are "full" widths, not radius
    #width, height = 2 * nstd * np.sqrt(vals)
    width = a
    height = b
    print(a, b, theta)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta)
    ellip.set_facecolor(color)
    ax.add_artist(ellip)
    return ellip


filepath_a = 'data1_a.txt'
filepath_c = 'data1_c.txt'
filepath_d = 'data1_d.txt'
observation_a = []
observation_d = []
known_c = []

num = 0
n = 9
n0 = 6
r = 3

with open(filepath_a) as fp:
    for line in fp:
        values = line.split()
        a_d = float(values[0])
        a_m = float(values[1])
        a_s = float(values[2])
        
        a = a_d / 180 * np.pi + a_m / 180 / 60 * np.pi + a_s / 180 / 3600 * np.pi
        
        observation_a.append([a])

observation_a = np.asarray(observation_a, dtype=np.float32)

with open(filepath_d) as fp:
    for line in fp:
        values = line.split()
        d = float(values[0])
        observation_d.append(d)

observation_d = np.asarray(observation_d, dtype=np.float32)


with open(filepath_c) as fp:
    for line in fp:
        values = line.split()
        x = float(values[0])
        y = float(values[1])
        known_c.append([x, y])

known_c = np.asarray(known_c, dtype=np.float32)

# Initial value:
Coordinate_Unknown_Ini = np.zeros([3, 2])
A_estimate = np.zeros([5, 1])
A_estimate[0] = np.arctan2(known_c[1][0] - known_c[0][0], known_c[1][1] - known_c[0][1])

x_p = known_c[1][0]
y_p = known_c[1][1]


for i in range(3):
    A_estimate[i + 1] = A_estimate[i] +  observation_a[i] - np.pi
    
    #if A_estimate[i + 1] > np.pi:
        #A_estimate[i + 1] = A_estimate[i + 1] - np.pi * 2
    
    Coordinate_Unknown_Ini[i][0] = np.sin(A_estimate[i + 1]) * observation_d[i] + x_p
    Coordinate_Unknown_Ini[i][1] = np.cos(A_estimate[i + 1]) * observation_d[i] + y_p
    
    x_p = Coordinate_Unknown_Ini[i][0]
    y_p = Coordinate_Unknown_Ini[i][1]

x3 = 1751 #Coordinate_Unknown_Ini[0][0]
y3 = 332#Coordinate_Unknown_Ini[0][1]
x4 = 1796#Coordinate_Unknown_Ini[1][0]
y4 = 394#Coordinate_Unknown_Ini[1][1]
x5 = 1844#Coordinate_Unknown_Ini[2][0]
y5 = 316#Coordinate_Unknown_Ini[2][1]



xcp1 = known_c[0][0]
ycp1 = known_c[0][1]
xcp2 = known_c[1][0]
ycp2 = known_c[1][1]
xcp6 = known_c[2][0]
ycp6 = known_c[2][1]
xcp7 = known_c[3][0]
ycp7 = known_c[3][1]


# weight:
w = np.zeros([9, 9])
sigma_0 = 0.015
sigma_a = 55 / 3600 / 180 * np.pi
sigma_d = sigma_0

for i in range(4):
    w[i][i] = sigma_0**2 / sigma_d ** 2
    
for i in range(5):
    w[i + 4][i + 4] = sigma_0**2 / sigma_a ** 2
    
num = 0
# Indirect observation:
while True:
    B = np.zeros([9, 6])
    f = np.zeros([9, 1])
    
    # observations:
    d10 = np.sqrt((x3 - xcp2) ** 2 + (y3 - ycp2) ** 2 )
    f[0][0] = d10 - observation_d[0]
    d20 = np.sqrt((x3 - x4) ** 2 + (y3 - y4) ** 2 )
    f[1][0] = d20 - observation_d[1]
    d30 = np.sqrt((x4 - x5) ** 2 + (y4 - y5) ** 2 )
    f[2][0] = d30 - observation_d[2]
    d40 = np.sqrt((x5 - known_c[2][0]) ** 2 + (y5 - ycp6) ** 2 )
    f[3][0] = d40 - observation_d[3]  
    
    r1 = (x3 - xcp2) / (y3 - ycp2)
    r2 = (x3 - x4) / (y3 - y4)
    r3 = (x4 - x5) / (y4 - y5)
    r4 = (x5 - xcp6) / (y5 - ycp6)
    
    f[4][0] = np.arctan2((x3 - xcp2), (y3 - ycp2)) - np.arctan2((xcp1 - xcp2), (ycp1 - ycp2)) - observation_a[0]
    f[5][0] = np.arctan2((x4 - x3), (y4 - y3)) - np.arctan2((xcp2 - x3), (ycp2 - y3)) - observation_a[1]
    f[6][0] = np.arctan2((x5 - x4), (y5 - y4)) - np.arctan2((x3 - x4), (y3 - y4)) - observation_a[2]
    f[7][0] = np.arctan2((xcp6 - x5), (ycp6 - y5)) - np.arctan2((x4 - x5), (y4 - y5)) - observation_a[3]
    f[8][0] = np.arctan2((xcp7 - xcp6), (ycp7 - ycp6)) - np.arctan2((x5 - xcp6), (y5 - ycp6)) - observation_a[4]
    
    f = - f
    
    B[0][0] = (x3 - xcp2) / d10
    B[0][1] = (y3 - ycp2) / d10
    
    B[1][0] = (x3 - x4) / d20
    B[1][1] = (y3 - y4) / d20
    B[1][2] = - B[1][0]
    B[1][3] = - B[1][1]
    
    B[2][2] = (x4 - x5) / d30
    B[2][3] = (y4 - y5) / d30
    B[2][4] = - B[2][2]
    B[2][5] = - B[2][3]
    
    B[3][4] = (x5 - xcp6) / d40
    B[3][5] = (y5 - ycp6) / d40
    
    
    B[4][0] = 1/(1 + r1**2) * 1 / (y3 - ycp2)
    B[4][1] = -1 / (1 + r1**2) * (x3 - xcp2) / (y3 - ycp2) ** 2 
    
    B[5][0] = -1 / (1 + r2**2) / (y4 - y3) - 1/(1 + r1**2) / (y3 - ycp2)
    B[5][1] = -1 / (1 + r2**2) * (x4 - x3) / (y4 - y3) ** 2 + 1 / (1 + r1**2) * (xcp2 - x3) / (y3 - ycp2) ** 2
    
    B[5][1] = - B[5][1]
    
    B[5][2] = 1/(1 + r2**2) * 1 / (y4 - y3)
    B[5][3] = - 1 / (1 + r2**2) * (x4 - x3) / (y4 - y3) ** 2
    
    B[6][0] = B[5][2]
    B[6][1] = B[5][3]
    B[6][2] = - 1 / (1 + r3**2) / (y5 - y4) + 1 / (1 + r2**2) / (y3 - y4)
    B[6][3] = - 1 / (1 + r3**2) * (x5 - x4) / (y5 - y4) ** 2 + 1 / (1 + r2**2) * (x3 - x4)/ (y3 - y4) ** 2
    
    B[6][3] = - B[6][3]
    
    B[6][4] = 1 / (1 + r3**2) / (y5 - y4)
    B[6][5] = -1 / (1 + r3**2) * (x5 - x4) / (y5 - y4) ** 2
    
    B[7][2] = -1 / (1 + r3**2) / (y4 - y5)
    B[7][3] = 1 / (1 + r3**2) * (x4 - x5) / (y4 - y5) ** 2
    B[7][4] = -1 / (1 + r4**2) / (ycp6 - y5) + 1/(1 + r3**2) / (y4 - y5)
    B[7][5] = 1 / (1 + r4**2) * (xcp6 - x5) / (ycp6 - y5) ** 2 - 1 / (1 + r3**2) * (x4 - x5) / (y4 - y5) ** 2
    
    B[8][4] = -1 / (1 + r4**2) / (y5 - ycp6)
    B[8][5] = 1 / (1 + r4**2) * (x5 - xcp6) / (y5 - ycp6) ** 2
    
    
    N = np.dot(np.dot(B.T,w), B)
    Q = np.linalg.inv(w)
    Q_delta = np.linalg.inv(N)
    x = np.dot(np.linalg.inv(np.dot(np.dot(B.T,w), B)), np.dot(np.dot(B.T,w), f))
    
    v = np.dot(B, x) - f
    
    num = num + 1
    
    x3 = x3 + x[0]
    y3 = y3 + x[1]
    x4 = x4 + x[2]
    y4 = y4 + x[3]
    x5 = x5 + x[4]
    y5 = y5 + x[5]
    
    x_abs = np.abs(x)
    x_abs_max = np.max(x_abs)
    print(x_abs_max)
    
    if num > 30 or x_abs_max < 0.0001:
        break
Naa = Q
danwei = np.eye(9)
Qkk = np.dot(np.linalg.inv(Naa), (danwei - np.dot(np.dot(B, np.linalg.inv(N)), np.dot(B.T, np.linalg.inv(Naa)))))
Qvv = np.dot(np.dot(Q, Qkk), Qkk)
WW = np.dot(Qvv, w)
 # Two sides global test
alpha = 0.05
posterio_sigma = np.sqrt(np.dot(np.dot(v.T,w), v) / r)
P1 = 0.025
P2 = 0.975
GlobalTest = np.dot(np.dot(v.T,w), v) / sigma_0 / sigma_0 

from scipy.stats import chi2, f

CV1 = chi2.ppf(q=0.025, df = r) # critical value 
CV2 = chi2.ppf(q=0.975, df = r)
#Accept the null hypothesis

# x3 and y3
Q_delta_3 = Q_delta[0:2,0:2]
Q_delta_4 = Q_delta[2:4,2:4]
Q_delta_5 = Q_delta[4:6,4:6]

Sigma_delta_3 = sigma_0 ** 2 * Q_delta_3 * 10000
Sigma_delta_4 = sigma_0 ** 2 * Q_delta_4 * 10000
Sigma_delta_5 = sigma_0 ** 2 * Q_delta_5 * 10000

Sigma_delta_3_p = posterio_sigma **2  * Q_delta_3 * 10000
Sigma_delta_4_p = posterio_sigma **2 * Q_delta_4 * 10000
Sigma_delta_5_p = posterio_sigma **2 * Q_delta_5 * 10000

fig, ax = plt.subplots(1, 1, figsize=(18, 12))

# plot the survey
ax.plot(xcp1, ycp1, 'k^')
ax.annotate('CP1({:.3f}, {:.3f})'.format(xcp1, ycp1), xy=(xcp1, ycp1 + 2),fontsize=16)
ax.plot(xcp2, ycp2, 'k^')
ax.annotate('CP2({:.3f}, {:.3f})'.format(xcp2, ycp2), xy=(xcp2, ycp2 + 2),fontsize=16)
ax.plot(xcp6, ycp6, 'k^')
ax.annotate('CP6({:.3f}, {:.3f})'.format(xcp6, ycp6), xy=(xcp6 - 5, ycp6 + 2),fontsize=16)
ax.plot(xcp7, ycp7, 'k^')
ax.annotate('CP7({:.3f}, {:.3f})'.format(xcp7, ycp7), xy=(xcp7, ycp7 + 2),fontsize=16)
ax.plot(x3, y3, 'ro')
ax.annotate('P3({:.3f}, {:.3f})'.format(x3[0], y3[0]), xy=(x3[0]+ 2, y3[0]),fontsize=16)
ax.plot(x4, y4, 'ro')
ax.annotate('P4({:.3f}, {:.3f})'.format(x4[0], y4[0]), xy=(x4[0], y4[0] + 2),fontsize=16)
ax.plot(x5, y5, 'ro')
ax.annotate('P5({:.3f}, {:.3f})'.format(x5[0], y5[0]), xy=(x5[0]+ 5, y5[0]),fontsize=16)

Cp2_chi2 = chi2.ppf(q = 0.99, df = 2)
Cp2_f =  2 * f.ppf(q = 0.99, dfn = 2, dfd = r) # F function

# 
plot_cov_ellipse(Sigma_delta_3_p, [x3, y3], ax, Cp2_f, 'yellow',nstd = 1)  
plot_cov_ellipse(Sigma_delta_4_p, [x4, y4], ax, Cp2_f, 'yellow',nstd = 1) 
plot_cov_ellipse(Sigma_delta_5_p, [x5, y5], ax, Cp2_f, 'yellow', nstd = 1) 
yellow_patch = mpatches.Patch(color='yellow', label='The H1 hypothesis confidence ellipse''s scale to 1 meter = 1 : 100')

# 
plot_cov_ellipse(Sigma_delta_3, [x3, y3], ax, Cp2_chi2, 'green', nstd = 1)  
plot_cov_ellipse(Sigma_delta_4, [x4, y4], ax, Cp2_chi2, 'green',nstd = 1) 
plot_cov_ellipse(Sigma_delta_5, [x5, y5], ax, Cp2_chi2, 'green',nstd = 1) 
green_patch = mpatches.Patch(color='green', label='The H0 hypothesis confidence ellipse''s scale to 1 meter = 1 : 100')


plt.plot([xcp1, xcp2], [ycp1, ycp2], 'b-')
plt.plot([xcp2, x3], [ycp2, y3], 'b-')
plt.plot([x3, x4], [y3, y4], 'b-')
plt.plot([x4, x5], [y4, y5], 'b-')
plt.plot([x5, xcp6], [y5, ycp6], 'b-')
plt.plot([xcp6, xcp7], [ycp6, ycp7], 'b-')

plt.legend(handles=[green_patch, yellow_patch], prop={'size': 16})
plt.title('Point Layout and the Survey', fontsize=20)