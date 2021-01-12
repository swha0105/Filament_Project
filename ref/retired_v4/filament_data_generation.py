#%%
import copy
import os
import sys
import pandas as pd 
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import shutil

#%%
class data_gen:

    def __init__(self,num_filament):
        self.num_filament = num_filament
        self.filament_list = {}

        for i in range(1,num_filament+1):
            self.filament_list[str(i)] = {}
            self.filament_list[str(i)]['coords'] = []
            self.filament_list[str(i)]['length'] = []
            self.filament_list[str(i)]['mean_dens'] = []
            self.filament_list[str(i)]['curvature'] = []



def sphereFit(spX,spY,spZ):
    #   Assemble the A matrix
    spX = np.array(spX)
    spY = np.array(spY)
    spZ = np.array(spZ)
    A = np.zeros((len(spX),4))
    A[:,0] = spX*2
    A[:,1] = spY*2
    A[:,2] = spZ*2
    A[:,3] = 1

    #   Assemble the f matrix
    f = np.zeros((len(spX),1))
    f[:,0] = (spX*spX) + (spY*spY) + (spZ*spZ)
    C, residules, rank, singval = np.linalg.lstsq(A,f)

    #   solve for the radius
    t = (C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2])+C[3]
    radius = math.sqrt(t)

    return radius, C[0], C[1], C[2]
#%%


box_length = '300Mpc'
resolution = 1024
box_num = '2'
distance = 0 

if box_length == '200Mpc':
    if resolution == 1024:
        distance = 100
    elif resolution == 512:
        distance = 50
elif box_length == '300Mpc':
    if resolution == 1024:
        distance = 70
    elif resolution == 512:
        distance = 35
elif box_length == '100Mpc':
    if resolution == 1024:
        distance = 200
    elif resolution == 512:
        distance = 100


data_path = '/storage/filament/works_v2/data_filament/' + box_length + '_' + box_num + '_' + str(resolution) + '/clusters/'
result_path = '/storage/filament/works_v2/result_filament/'  + box_length + '_' + box_num + '_' + str(resolution) + '/'
 

res = distance*2 + 1 
box_size = int(distance/10)
center_point = distance


ref_path = '/storage/filament/works_v2/result_filament/' 


df = pd.read_csv(ref_path + box_length + '_' + box_num + '_' + str(resolution) + '_' + 'info/'  + 'virgo_info.csv')
virgo_info = df.to_numpy()[:,1]


df = pd.read_csv(ref_path + box_length + '_' + box_num + '_' + str(resolution) + '_' + 'info/'  +  'clusters_info.csv')
cluster_coords = df.to_numpy()[:,1:]



filament_info = {}

for num_cluster in virgo_info:
#for num_cluster in range(82,83):
    cluster_path = result_path   + str(num_cluster) + '/'
    filament_path = cluster_path + 'filament/'
    dens_path = data_path + str(num_cluster) + '/'

    filament_pre_list = []
    filament_del_list = [ ]

    for filament_num in range(1,len(os.listdir(filament_path))+1):
        tmp = np.genfromtxt(filament_path + str(filament_num), dtype='i4')
        if len(tmp) >= int( 5/(int(box_length[:3])/resolution )):
            filament_pre_list.append(np.genfromtxt(filament_path + str(filament_num), dtype='i4' ))
        else:
            continue
        

    for n in range(len(filament_pre_list)-1):
        ref_filament = filament_pre_list[n]
        for m in range(n+1,len(filament_pre_list)):
            compare_filament = filament_pre_list[m]

            count = 0 

            for nn in range(len(compare_filament)):
                ref_x = ref_filament[nn,0]
                ref_y = ref_filament[nn,1]
                ref_z = ref_filament[nn,2]

                for mm in range(len(compare_filament)):
                    compare_x = compare_filament[mm,0]
                    compare_y = compare_filament[mm,1]
                    compare_z = compare_filament[mm,2]

                    if ref_x == compare_x and ref_y == compare_y and ref_z == compare_z:
                        count = count + 1 
                        break 
                    else:
                        continue
            
            if count/len(compare_filament) > 0.3:
                filament_del_list.append(m)
                pass

    
    if filament_del_list == []:
        filament_sort = filament_pre_list
    else:
        filament_sort = np.delete(filament_pre_list,filament_del_list)
    

    filament = data_gen(len(filament_sort))

    for num in range(len(filament_sort)):
        filament.filament_list[str(num+1)]['coords'] = filament_sort[num]




    dens = np.load(dens_path + 'dens.npy').reshape([res,res,res])
    
    
    for num in range(len(filament_sort)):
        dens_sum = 0 

        for n,(ix,iy,iz) in enumerate(filament_sort[num]):
            dens_sum = dens_sum + dens[ix,iy,iz] 

        dens_log_mean = np.log10(dens_sum/len(filament_sort[num]))
        
        filament.filament_list[str(num+1)]['mean_dens'] = dens_log_mean
        filament.filament_list[str(num+1)]['length'] = len(filament_sort[num])



    # for num in range(len(filament_sort)):
    #     dens_sum = 0 

    #     coords = filament.filament_list[str(num+1)]['coords']
        
    #     for i in range(len(coords)-5):
    #         x_coords = coords[i:i+5,0]   
    #         y_coords = coords[i:i+5,1]
    #         z_coords = coords[i:i+5,2]

    #         r,_,_,_ = sphereFit(x_coords,y_coords,z_coords)
    #         radius = r*(int(box_length[:3])/resolution)
    #         curvature = 1/radius
            
    #         filament.filament_list[str(num+1)]['curvature'].append(curvature)
            

            
    length = []
    mean_dens = []


    save_path = cluster_path + 'filament_info/'
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    if not os.path.isdir(save_path):        
        os.makedirs(save_path)     

    for num in range(len(filament.filament_list)):

        coords_path = save_path + 'coords/'
        curvature_path = save_path + 'curvature/'

        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        if not os.path.isdir(coords_path):
            os.makedirs(coords_path)
        if not os.path.isdir(curvature_path):
            os.makedirs(curvature_path)

        coords = filament.filament_list[str(num+1)]['coords']
        #curvature = filament.filament_list[str(num+1)]['curvature']
        length.append(filament.filament_list[str(num+1)]['length'])
        mean_dens.append(filament.filament_list[str(num+1)]['mean_dens'])
        
        

        np.savetxt(coords_path + str(num+1), coords,fmt='%3i')
        #np.savetxt(curvature_path + str(num+1), curvature,fmt='%1.5f')



    df = pd.DataFrame({'mean_dens (log)': mean_dens, 'length (grid)': length}, index= [np.linspace(1,len(filament.filament_list),len(filament.filament_list),dtype='i4')] )
    df.to_csv(save_path + 'info.csv')

  
    

#%%
