#%%
import copy
import os
import sys
import shutil
import cv2 
import matplotlib.pyplot as plt
import numpy as np
import copy



def get_F(ix,iy,iz,cost,end):
    #G = np.sqrt(  (ix-start[0])**2 + (iy-start[1])**2 + (iz-start[2])**2 )
    G = cost
    H = np.sqrt(  (ix-end[0])**2 + (iy-end[1])**2 + (iz-end[2])**2 )*10

    F = G+H
    #print(F,G,H)
    return F

    
def get_note(array,current_position,cost_sum,end):
    
    c_x = current_position[0]
    c_y = current_position[1]
    c_z = current_position[2]

    min_x = c_x - 1
    max_x = c_x + 2

    if min_x < 0:
        min_x = 0
    if max_x > array.shape[0]-1:
        max_x = array.shape[0]-1
    
    min_y = c_y - 1
    max_y = c_y + 2

    if min_y < 0:
        min_y = 0
    if max_y > array.shape[0]-1:
        max_y = array.shape[0]-1

    min_z = c_z - 1
    max_z = c_z + 2

    if min_z < 0:
        min_z = 0
    if max_z > array.shape[0]-1:
        max_z = array.shape[0]-1
    
    ix_list = []
    iy_list = []
    iz_list = []

    candidate_coords = np.argwhere(array[min_x:max_x,min_y:max_y,min_z:max_z]!=0)
    
    candidate_coords = candidate_coords - 1
    
    tmp_F = []
    for ix,iy,iz in candidate_coords:
        new_x = ix+c_x
        new_y = iy+c_y
        new_z = iz+c_z

        if new_x < 0:
            continue
        if new_y < 0:
            continue
        if new_z < 0:
            continue
        
        if ix == 0 and iy == 0 and iz == 0:
            continue
        else:


            tmp_F.append(get_F(new_x,new_y,new_z,cost_sum,end))
            
            ix_list.append(new_x)
            iy_list.append(new_y)
            iz_list.append(new_z)

    #print(tmp_F)
    #print(iz_list)
    
    break_argue = 0 
    if not tmp_F:
        break_argue = 1
        return _,_,break_argue
    
    tmp_idx = np.argmin(np.array(tmp_F))
    ix_n = ix_list[tmp_idx]
    iy_n = iy_list[tmp_idx]
    iz_n = iz_list[tmp_idx]
    cost=10
    if np.abs(ix)+np.abs(iy)+np.abs(iz) == 3:
        cost = 27
    elif np.abs(ix)+np.abs(iy)+np.abs(iz) == 2:
        cost = 14
    elif np.abs(ix)+np.abs(iy)+np.abs(iz) == 1:
        cost = 10
    
    
    cost_sum = cost_sum + cost

    coords = [ix_n,iy_n,iz_n]
    #print(cost_sum)

    return coords,cost_sum,break_argue
#%%

ref_path = '/storage/filament/works_v5/300Mpc_1/clusters/label/'
skeleton_path = ref_path + 'skeleton/'

 
endpoint_array = np.linspace(36-3,36+3,7)
endpoint = [36,36,36]
for cluster_num in np.sort(np.array(os.listdir(skeleton_path))):

    skeleton = np.load(skeleton_path + cluster_num)

    print(cluster_num)
    filament_start_point = np.array(np.where(skeleton==2)).T
    filament_end_point = [int(skeleton.shape[0]/2),int(skeleton.shape[0]/2),int(skeleton.shape[0]/2)]


    num = 1
    for num_filament in range(filament_start_point.shape[0]):
        
        skeleton_open = copy.deepcopy(skeleton)

        startpoint = filament_start_point[num_filament,:]
        #startpoint = [25,27,0]
        current_coords = startpoint


        filament_coords = []
        filament_coords.append(current_coords)

        cost_sum = 0
        break_arg = 0
        while break_arg==0:
            current_coords,cost_sum,break_arg = get_note(skeleton_open,current_coords,cost_sum,endpoint)
            
            if break_arg  == 1:
                break

            
            filament_coords.append(current_coords)
            skeleton_open[current_coords[0],current_coords[1],current_coords[2]] = 0
            if (np.array(current_coords)[0] in endpoint_array and np.array(current_coords)[1] in endpoint_array and np.array(current_coords)[2] in endpoint_array ):
                
                if not os.path.isdir(ref_path + 'segment/' + cluster_num +'/'):
                    os.makedirs(ref_path + 'segment/' + cluster_num +'/')
                np.savetxt(ref_path + 'segment/' + cluster_num +'/' + str(num) ,filament_coords)
                num = num + 1
                break

# %%






#%%









endpoint = filament_end_point[0,:]

current_coords = filament_start_point
startpoint = filament_start_point
filament_coords = []
filament_coords.append(current_coords)
while True:
    current_coords = get_note(skeleton_open,current_coords,startpoint,endpoint)
    filament_coords.append(current_coords)
    skeleton_open[current_coords[0],current_coords[1],current_coords[2]] = 0
    if (np.array(current_coords) == endpoint).all():
        break