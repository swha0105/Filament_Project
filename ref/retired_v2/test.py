#%%

import copy
import os
import sys
import shutil
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

#%%
A = np.genfromtxt('/storage/filament/works_v2/result_filament/200Mpc_1_1024/1/filament_post_2/filament_info/coords/1',dtype='i4')

# %%
smooth_scale = 10
box_length = '200Mpc'
resolution = 1024
coords = A
curvature_list = []
len_list = []
for i in range(len(coords)-smooth_scale):
#for i in range(0,1):
    x_coords = coords[i:i+smooth_scale,0]   
    y_coords = coords[i:i+smooth_scale,1]
    z_coords = coords[i:i+smooth_scale,2]

    r,_,_,_ = sphereFit(x_coords,y_coords,z_coords)
    radius = r*(int(box_length[:3])/resolution)
    curvature = 1.0/radius
    curvature_list.append(curvature)
    len_list.append(i)


# %%
curv_mean_2 = np.mean(curvature_list)
#%%
curv_mean_4 = np.mean(curvature_list)
#%%
curv_mean_6 = np.mean(curvature_list)

#%%
plt.plot(len_list,curvature_list)
plt.xlabel('len (grid)',fontsize=15)
plt.ylabel('curvature',fontsize=15)

#%%

#-------------------------------------------------------------------------------
# FIT CIRCLE 2D
# - Find center [xc, yc] and radius r of circle fitting to set of 2D points
# - Optionally specify weights for points
#
# - Implicit circle function:
#   (x-xc)^2 + (y-yc)^2 = r^2
#   (2*xc)*x + (2*yc)*y + (r^2-xc^2-yc^2) = x^2+y^2
#   c[0]*x + c[1]*y + c[2] = x^2+y^2
#
# - Solution by method of least squares:
#   A*c = b, c' = argmin(||A*c - b||^2)
#   A = [x y 1], b = [x^2+y^2]
#-------------------------------------------------------------------------------
def fit_circle_2d(x, y, w=[]):
    
    A = np.array([x, y, np.ones(len(x))]).T
    b = x**2 + y**2
    
    # Modify A,b for weighted least squares
    if len(w) == len(x):
        W = np.diag(w)
        A = np.dot(W,A)
        b = np.dot(W,b)
    
    # Solve by method of least squares
    c = np.linalg.lstsq(A,b,rcond=None)[0]
    
    # Get circle parameters from solution c
    xc = c[0]/2
    yc = c[1]/2
    r = np.sqrt(c[2] + xc**2 + yc**2)
    return xc, yc, r


#-------------------------------------------------------------------------------
# RODRIGUES ROTATION
# - Rotate given points based on a starting and ending vector
# - Axis k and angle of rotation theta given by vectors n0,n1
#   P_rot = P*cos(theta) + (k x P)*sin(theta) + k*<k,P>*(1-cos(theta))
#-------------------------------------------------------------------------------
def rodrigues_rot(P, n0, n1):
    
    # If P is only 1d array (coords of single point), fix it to be matrix
    if P.ndim == 1:
        P = P[newaxis,:]
    
    # Get vector of rotation k and angle theta
    n0 = n0/np.linalg.norm(n0)
    n1 = n1/np.linalg.norm(n1)
    k = np.cross(n0,n1)
    k = k/np.linalg.norm(k)
    theta = np.arccos(np.dot(n0,n1))
    
    # Compute rotated points
    P_rot = np.zeros((len(P),3))
    for i in range(len(P)):
        P_rot[i] = P[i]*np.cos(theta) + np.cross(k,P[i])*np.sin(theta) + k*np.dot(k,P[i])*(1-np.cos(theta))

    return P_rot


#-------------------------------------------------------------------------------
# ANGLE BETWEEN
# - Get angle between vectors u,v with sign based on plane with unit normal n
#-------------------------------------------------------------------------------
def angle_between(u, v, n=None):
    if n is None:
        return arctan2(np.linalg.norm(np.cross(u,v)), np.dot(u,v))
    else:
        return arctan2(np.dot(n,np.cross(u,v)), np.dot(u,v))

    
#-------------------------------------------------------------------------------
# - Make axes of 3D plot to have equal scales
# - This is a workaround to Matplotlib's set_aspect('equal') and axis('equal')
#   which were not working for 3D
#-------------------------------------------------------------------------------
def set_axes_equal_3d(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    spans = abs(limits[:,0] - limits[:,1])
    centers = mean(limits, axis=1)
    radius = 0.5 * max(spans)
    ax.set_xlim3d([centers[0]-radius, centers[0]+radius])
    ax.set_ylim3d([centers[1]-radius, centers[1]+radius])
    ax.set_zlim3d([centers[2]-radius, centers[2]+radius])

#%%
smooth_scale = 10

P = coords[0:10,:]
P_mean = P.mean(axis=0)
P_centered = P - P_mean
U,s,V = np.linalg.svd(P_centered)

# Normal vector of fitting plane is given by 3rd column in V
# Note linalg.svd returns V^T, so we need to select 3rd row from V^T
normal = V[2,:]
d = -np.dot(P_mean, normal)  # d = -<p,n>
alpha_pts = 0.5

#-------------------------------------------------------------------------------
# (2) Project points to coords X-Y in 2D plane
#-------------------------------------------------------------------------------
P_xy = rodrigues_rot(P_centered, normal, [0,0,1])
#plt.figure(figsize=[10,10])
#plt.scatter(P_xy[:,0], P_xy[:,1], alpha=alpha_pts, label='Projected points')

#-------------------------------------------------------------------------------
# (3) Fit circle in new 2D coords
#-------------------------------------------------------------------------------
xc, yc, r = fit_circle_2d(P_xy[:,0], P_xy[:,1])

print(1/(r*0.2))




#%%

smooth_scale = 10
curvature_list = []

for j in range(1,4):
    j = str(j)
    A = np.genfromtxt('/storage/filament/works_v2/result_filament/200Mpc_1_1024/1/filament_post_2/filament_info/coords/' + j,dtype='i4')
    curvature = []

    for i in range(len(A)-smooth_scale):
        P = A[i:i+smooth_scale,:]

        
        if (P[0,2] == P[:,2]).all():

        else:
            P_mean = P.mean(axis=0)
            P_centered = P - P_mean
            U,s,V = np.linalg.svd(P_centered)

    
            normal = V[2,:]
            d = -np.dot(P_mean, normal)  
            alpha_pts = 0.5

            P_xy = rodrigues_rot(P_centered, normal, [0,0,1])

            xc, yc, r = fit_circle_2d(P_xy[:,0], P_xy[:,1])
            curvature.append(1.0/(r*0.2))
        

    curvature_list.append(curvature)
        

    #%%

curvature_list = np.array(curvature_list)
# %%
plt.plot(curvature_list[2][:])   

# %%
x = P[:,0]
y = P[:,1]
##2D
#%%

def circular_fitting_2d(x,y):
    # coordinates of the barycenter
    x_m = np.mean(x)
    y_m = np.mean(y)

    # calculation of the reduced coordinates
    u = x - x_m
    v = y - y_m

    # linear system defining the center (uc, vc) in reduced coordinates:
    #    Suu * uc +  Suv * vc = (Suuu + Suvv)/2
    #    Suv * uc +  Svv * vc = (Suuv + Svvv)/2
    Suv  = np.sum(u*v)
    Suu  = np.sum(u**2)
    Svv  = np.sum(v**2)
    Suuv = np.sum(u**2 * v)
    Suvv = np.sum(u * v**2)
    Suuu = np.sum(u**3)
    Svvv = np.sum(v**3)

    # Solving the linear system
    A = np.array([ [ Suu, Suv ], [Suv, Svv]])
    B = np.array([ Suuu + Suvv, Svvv + Suuv ])/2.0
    uc, vc = np.linalg.solve(A, B)

    xc_1 = x_m + uc
    yc_1 = y_m + vc

    # Calcul des distances au centre (xc_1, yc_1)
    Ri_1     = np.sqrt((x-xc_1)**2 + (y-yc_1)**2)
    R_1      = np.mean(Ri_1)
    residu_1 = np.sum((Ri_1-R_1)**2)
    
    return R_1