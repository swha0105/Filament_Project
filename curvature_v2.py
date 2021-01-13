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
import shutil


def first_diff(coef,tn):
    val = 3*(coef[0]**2)*(tn**2) + 2*coef[1]*tn + coef[2]
    return val

def second_diff(coef,tn):
    val = 6*(coef[0])*(tn) + 2*coef[1]
    return val



box_length = '300Mpc'
box_num = 1
ref_path = '/storage/filament/works_v7/' + box_length + '_' + str(box_num) + '/filament/'
mpc_grid = int(box_length[:3])/2048
#%%

        
if box_length == '200Mpc':
    cluster_grid = 200*2 + 1
if box_length == '300Mpc':
    cluster_grid = 135*2 + 1


for cluster_num in np.sort(os.listdir(ref_path + 'sorted/')):
    cluster_num = '22'
    cluster_path = ref_path + 'sorted/' + str(cluster_num) + '/'
    fitting_path = ref_path + 'fitting/' + str(cluster_num) + '/'
    curvature_path = ref_path + 'curvature/' + str(cluster_num) + '/'

    if not os.path.isdir(fitting_path):
        os.makedirs(fitting_path)

    if not os.path.isdir(curvature_path):
        os.makedirs(curvature_path)

    if os.path.isdir(fitting_path):
       shutil.rmtree(fitting_path)
    os.makedirs(fitting_path)
    
    if os.path.isdir(curvature_path):
       shutil.rmtree(curvature_path)
    os.makedirs(curvature_path)
    
    filament_sort_count = 0    
    num = 1 
    for filament_num in np.sort(os.listdir(cluster_path)):
        
        filament = np.loadtxt(cluster_path + str(filament_num))
        
        # if filament.shape[0] < int(5/mpc_grid):
        #     continue
        # 3-order fitting 
        # if len(filament) <= int(2/mpc_grid): #5Mpc
        #     continue
        # else:
        filament_sort_count = filament_sort_count + 1
        coords_fit = []
        length = np.linspace(1,len(filament),len(filament))*mpc_grid

        fitx = np.polyfit(length, filament[:,0], 3)
        fity = np.polyfit(length, filament[:,1], 3)
        fitz = np.polyfit(length, filament[:,2], 3)

        tmpx = np.polyval([fitx[0],fitx[1],fitx[2],fitx[3]], length)
        tmpy = np.polyval([fity[0],fity[1],fity[2],fity[3]], length)
        tmpz = np.polyval([fitz[0],fitz[1],fitz[2],fitz[3]], length)

        x_fit = []
        y_fit = []
        z_fit = []
        #print(filament_num,len(filament),len(length))
        for j in range(len(length)):

            if tmpx[j] > cluster_grid-1:
                tmpx[j]= cluster_grid-1
            if tmpx[j] <= 0 :
                tmpx[0]

            if tmpy[j] > cluster_grid-1:
                tmpy[j]= cluster_grid-1
            if tmpy[j] <= 0 :
                tmpy[0]

            if tmpz[j] > cluster_grid-1:
                tmpz[j]= cluster_grid-1
            if tmpz[j] <= 0 :
                tmpz[0]

            x_fit.append(int(round(tmpx[j])))
            y_fit.append(int(round(tmpy[j])))
            z_fit.append(int(round(tmpz[j])))

        filament_tmp = np.array(list(zip(x_fit,y_fit,z_fit)))

        
        np.savetxt(fitting_path + str(num),filament_tmp,fmt='%3i')

        curvature_list = []
        tn_range = np.linspace(0,len(filament_tmp),len(filament_tmp)+1)*mpc_grid
   
        for tn in tn_range:
            curvature = np.sqrt( (first_diff(fity,tn) * second_diff(fitz,tn) - first_diff(fitz,tn) * second_diff(fity,tn))**2 + \
            (first_diff(fitx,tn) * second_diff(fitz,tn) - first_diff(fitz,tn) * second_diff(fitx,tn))**2 + \
            (first_diff(fitx,tn) * second_diff(fity,tn) - first_diff(fity,tn) * second_diff(fitx,tn))**2 \
            )/ (np.sqrt( first_diff(fitx,tn)**2 + first_diff(fity,tn)**2 + first_diff(fitz,tn)**2  ) )**3          
            
            curvature_list.append(curvature)

        
        np.savetxt(curvature_path + str(num),curvature_list,fmt='%1.5f')
        num = num + 1

    break


#%

# %%
