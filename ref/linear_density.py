#%%
import copy
import os
import sys
import shutil
import cv2 
import matplotlib.pyplot as plt
import numpy as np
import copy
from itertools import permutations,combinations,product
from numba import jit,njit, float32,complex64,int32

#%%
@jit(float32(float32,float32))
def dens_summation(density,coords):
    dens_sum = 0
    for num in range(len(coords)):
        dens_sum = dens_sum + density[coords[num]]
    return dens_sum


#@jit() 
@jit(float32(float32,float32))
def width_labeling(filament_skeleton,label):
    x_coords = []
    y_coords = []
    z_coords = []
    width = []
    current_coords = np.zeros(3,dtype=np.int32)
    for n in range(len(filament_skeleton)-1):
        
        #current_coords =  np.asarray(filament_skeleton[n],dtype=np.int32)
        current_coords[0] = int(filament_skeleton[n,0])
        current_coords[1] = int(filament_skeleton[n,1])
        current_coords[2] = int(filament_skeleton[n,2])
        if  (current_coords[0] < 150 + 13 and current_coords[0] > 150 - 13) and \
            (current_coords[1] < 150 + 13 and current_coords[1] > 150 - 13) and \
            (current_coords[2] < 150 + 13 and current_coords[2] > 150 - 13):
            continue
        
        for i in range(1,33):
            count = 0 
            for ix in range(-i,i+1):
                for iy in range(-i,i+1):
                    for iz in range(-i,i+1):
                        if np.sqrt((ix)**2 + (iy)**2 + (iz)** 2) < i:
                            if label[current_coords[0] + ix,current_coords[1] + iy,current_coords[2] + iz] == 1:
                                count = count + 1
                        
                                x_coords.append(current_coords[0] + ix )
                                y_coords.append(current_coords[1] + iy )
                                z_coords.append(current_coords[2] + iz )
            
            
            if count/int(4/3*np.pi*i**3) < 0.5 and i !=1:
                width.append(i)
                break

    coords = list(zip(x_coords,y_coords,z_coords))
    return coords,width
#%%
density_unit = (1.879e-29*(1)**3*(0.7**2)*0.044)*(((3.086e+24)/0.7)**3)/(2*10**33)

filament_ref_path = '/storage/filament/works_v7/300Mpc_1/filament/sorted/'
density_ref_path = '/storage/filament/works_v7/300Mpc_1/cluster_box/dens/'
label_ref_path = '/storage/filament/works_v7/300Mpc_1/label/upsampling/'
save_ref_path = '/storage/filament/works_v7/300Mpc_1/filament/whole/'

for cluster_num in np.sort(np.array(os.listdir(filament_ref_path))):
    filament = []
    density = 10**np.load(density_ref_path + cluster_num + '.npy')
    label = np.load(label_ref_path + cluster_num + '.npy')

    for filament_num in np.sort(np.array(os.listdir(filament_ref_path + cluster_num +'/'))):
        filament.append(np.loadtxt(filament_ref_path + cluster_num + '/' + filament_num))
    
    plane = np.zeros(3)
    dens_list = []
    for filament_num in range(len(np.array(filament))):
        print(cluster_num,filament_num)
        filament_skeleton = filament[filament_num] 
        coords = width_labeling(filament_skeleton,label)
        
        dens_sum = dens_summation(density,coords)
        dens_list.append(np.log10(dens_sum*density_unit/len(coords)))

        if not os.path.isdir(save_ref_path + cluster_num + '/'):
            os.makedirs(save_ref_path + cluster_num + '/')
        np.savetxt(save_ref_path + cluster_num + '/' + str(filament_num+1),coords,fmt='%i')    
    
    np.savetxt(save_ref_path + cluster_num + '/' + 'dens_list',dens_list)    
    


#%%
density_unit = (1.879e-29*(1)**3*(0.7**2)*0.044)/(2*10**33)

filament_ref_path = '/storage/filament/works_v7/300Mpc_1/filament/sorted/'
density_ref_path = '/storage/filament/works_v7/300Mpc_1/cluster_box/dens/'
label_ref_path = '/storage/filament/works_v7/300Mpc_1/label/upsampling/'
save_ref_path = '/storage/filament/works_v7/300Mpc_1/filament/whole/'

for cluster_num in np.sort(np.array(os.listdir(filament_ref_path)))[5:]:

    #cluster_num = '1'
    print(cluster_num)
    filament = []
    density = 10**np.load(density_ref_path + cluster_num + '.npy')
    label = np.load(label_ref_path + cluster_num + '.npy')

    for filament_num in np.sort(np.array(os.listdir(filament_ref_path + cluster_num +'/'))):
        filament.append(np.loadtxt(filament_ref_path + cluster_num + '/' + filament_num))
    
    plane = np.zeros(3)
    dens_list = []
    for filament_num in range(len(np.array(filament))):
        filament_skeleton = filament[filament_num] 
        x_coords = []
        y_coords = []
        z_coords = []
        width = []
        for n in range(len(filament_skeleton)-1):
            print(filament_num,n)
            #direction = filament_skeleton[n+1] - filament_skeleton[n]
            current_coords =  np.asarray(filament_skeleton[n],dtype=np.int32)

            if  (current_coords[0] < 150 + 13 and current_coords[0] > 150 - 13) and \
                (current_coords[1] < 150 + 13 and current_coords[1] > 150 - 13) and \
                (current_coords[2] < 150 + 13 and current_coords[2] > 150 - 13):
                continue
            
            for i in range(1,33):
                count = 0 
                for ix in range(-i,i+1):
                    for iy in range(-i,i+1):
                        for iz in range(-i,i+1):
                            if np.sqrt((ix)**2 + (iy)**2 + (iz)** 2) < i:
                                if label[current_coords[0] + ix,current_coords[1] + iy,current_coords[2] + iz] == 1:
                                    count = count + 1
                            
                                    x_coords.append(current_coords[0] + ix )
                                    y_coords.append(current_coords[1] + iy )
                                    z_coords.append(current_coords[2] + iz )
                
                
                if count/int(4/3*np.pi*i**3) < 0.5 and i !=1:
                    width.append(i)
                    break

        
        coords = list(zip(x_coords,y_coords,z_coords))
        if len(coords) == 0:
            continue
        else:
            dens_sum = 0
            
            dens_sum = dens_summation(density,coords)
            dens_list.append(np.log10(dens_sum*density_unit*((0.15*3*10**24)**3)))
            
            if not os.path.isdir(save_ref_path + cluster_num + '/'):
                os.makedirs(save_ref_path + cluster_num + '/')
            np.savetxt(save_ref_path + cluster_num + '/' + str(filament_num+1),coords,fmt='%i')    
            np.savetxt(save_ref_path + cluster_num + '/' + str(filament_num+1) + '_width',width,fmt='%i')    

    np.savetxt(save_ref_path + cluster_num + '/' + 'dens_list',dens_list)    
    
