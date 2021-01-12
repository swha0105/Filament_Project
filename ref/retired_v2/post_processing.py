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
from fitting_utils  import *

# %%
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

resolution = 1024
distance = 0 
box_length_list = ['200Mpc','300Mpc']
box_num_list = ['1','2']

for box_length in box_length_list:
    for box_num in box_num_list:
                
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

        res = 2*distance + 1

        minimum_length = int(distance/5)
        smooth_scale = int(distance/10) 


        data_path = '/storage/filament/works_v2/data_filament/' + box_length + '_' + box_num + '_' + str(resolution) + '/clusters/'
        result_path = '/storage/filament/works_v2/result_filament/'  + box_length + '_' + box_num + '_' + str(resolution) + '/'
        

        ref_path  = '/storage/filament/works_v2/result_filament/' + box_length + '_' + box_num + '_' + str(resolution) + '_info/' 

        df = pd.read_csv(ref_path + 'virgo_info.csv')
        virgo_info = df.to_numpy()[:,1]

 


        for cluster_num in virgo_info:


            filament_num_list = []
            filament_smooth_list = []

            filament_path = result_path + str(cluster_num) + '/filament_info/coords/'
            count = 0 
            if not os.path.isdir(filament_path):
                continue
            else:
                for filament_num in np.array(sorted(os.listdir(filament_path),key=int)):
                    

                    tmp_filament = np.genfromtxt(filament_path + str(filament_num), dtype='i4' )

                    length_filament = tmp_filament.shape[0]        

                    if length_filament < minimum_length:
                        continue
                    else:
                        count = count + 1 
                        x_smooth = []
                        y_smooth = []
                        z_smooth = []

                        for nn in range(length_filament-smooth_scale):
                            
                        
                            x_smooth.append(int(np.mean(tmp_filament[nn:nn+smooth_scale,0])))
                            y_smooth.append(int(np.mean(tmp_filament[nn:nn+smooth_scale,1])))
                            z_smooth.append(int(np.mean(tmp_filament[nn:nn+smooth_scale,2])))

                            filament_smooth = np.array(list(zip(x_smooth,y_smooth,z_smooth)))
                            
                    
                    filament_num_list.append(count)
                    filament_smooth_list.append(filament_smooth)

                save_path = result_path + str(cluster_num) + '/filament_post/'
                save_filament = sorted(filament_smooth_list,key=len,reverse=True)    
                
                if os.path.isdir(save_path):
                    shutil.rmtree(save_path)
                
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
            

                for num in filament_num_list:
                    np.savetxt(save_path + str(num), save_filament[num-1],fmt='%3i')


                coords_fit = []
                for i in range(len(save_filament)):
                    length = np.linspace(1,len(save_filament[i]),len(save_filament[i]))

                    fitx = np.polyfit(length, save_filament[i][:,0], 3)
                    fity = np.polyfit(length, save_filament[i][:,1], 3)
                    fitz = np.polyfit(length, save_filament[i][:,2], 3)

                    tmpx = np.polyval([fitx[0],fitx[1],fitx[2],fitx[3]], length)
                    tmpy = np.polyval([fity[0],fity[1],fity[2],fity[3]], length)
                    tmpz = np.polyval([fitz[0],fitz[1],fitz[2],fitz[3]], length)

                    x_fit = []
                    y_fit = []
                    z_fit = []

                    for j in range(len(length)):

                        if tmpx[j] > res-1:
                            tmpx[j]= res-1
                        if tmpx[j] <= 0 :
                            tmpx[0]

                        if tmpy[j] > res-1:
                            tmpy[j]= res-1
                        if tmpy[j] <= 0 :
                            tmpy[0]

                        if tmpz[j] > res-1:
                            tmpz[j]= res-1
                        if tmpz[j] <= 0 :
                            tmpz[0]

                        x_fit.append(int(round(tmpx[j])))
                        y_fit.append(int(round(tmpy[j])))
                        z_fit.append(int(round(tmpz[j])))

                    tmp = np.array(list(zip(x_fit,y_fit,z_fit)))
                    coords_fit.append(tmp)

                filament = data_gen(len(coords_fit))

                save_path = result_path + str(cluster_num) + '/filament_post_2/'

                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
                
                for num in range(len(coords_fit)):
                    np.savetxt(save_path + str(num+1), coords_fit[num],fmt='%3i')


                dens = np.load(data_path + str(cluster_num) + '/dens.npy').reshape([res,res,res])


                for num in range(len(coords_fit)):
                    dens_sum = 0 

                    for n,(ix,iy,iz) in enumerate(coords_fit[num]):
                        dens_sum = dens_sum + dens[ix,iy,iz] 

                    dens_log_mean = np.log10(dens_sum/len(coords_fit[num]))
            
                    filament.filament_list[str(num+1)]['mean_dens'] = dens_log_mean
                    filament.filament_list[str(num+1)]['length'] = len(coords_fit[num])



                for num in range(len(coords_fit)):
                
                    filament.filament_list[str(num+1)]['coords'] = coords_fit[num]
                    
                    coords = coords_fit[num]
            
                    for i in range(len(coords)-smooth_scale):
                        x_coords = coords[i:i+smooth_scale,0]   
                        y_coords = coords[i:i+smooth_scale,1]
                        z_coords = coords[i:i+smooth_scale,2]

                        r,_,_,_ = sphereFit(x_coords,y_coords,z_coords)
                        radius = r*(int(box_length[:3])/resolution)
                        curvature = 1.0/radius
                        
                        filament.filament_list[str(num+1)]['curvature'].append(curvature)
                
                
                length = []
                mean_dens = []

                save_path_2 = save_path + 'filament_info/'

                if os.path.isdir(save_path_2):        
                    shutil.rmtree(save_path_2)

                if not os.path.isdir(save_path_2):        
                    os.makedirs(save_path_2)     

                for num in range(len(filament.filament_list)):

                    coords_path = save_path_2 + 'coords/'
                    curvature_path = save_path_2 + 'curvature/'

                    if not os.path.isdir(save_path_2):
                        os.makedirs(save_path_2)
                    if not os.path.isdir(coords_path):
                        os.makedirs(coords_path)
                    if not os.path.isdir(curvature_path):
                        os.makedirs(curvature_path)

                    coords = filament.filament_list[str(num+1)]['coords']
                    curvature = filament.filament_list[str(num+1)]['curvature']
                    length.append(filament.filament_list[str(num+1)]['length'])
                    mean_dens.append(filament.filament_list[str(num+1)]['mean_dens'])
                    
                    

                    np.savetxt(coords_path + str(num+1), coords,fmt='%3i')
                    np.savetxt(curvature_path + str(num+1), curvature,fmt='%1.5f')



                df = pd.DataFrame({'mean_dens (log)': mean_dens, 'length (grid)': length}, index= [np.linspace(1,len(filament.filament_list),len(filament.filament_list),dtype='i4')] )
                df.to_csv(save_path_2 + 'info.csv')

        # %%
