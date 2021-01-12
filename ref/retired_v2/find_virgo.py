#%%
import copy
import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shutil
#%%

box_size = '300Mpc'
box_num = '2'
resolution = '1024'
ref_distance = 0 


ref_distance = int(10 / ( (int(box_size[:3])) / int(resolution) ) )

ref_path  = '/storage/filament/works_v2/result_filament/' + box_size + '_' + box_num + '_' + resolution + '_info/'
df = pd.read_csv(ref_path  + 'clusters_info.csv')

cluster_num = df.to_numpy(dtype='int')[:,1]
cluster_coords = df.to_numpy()[:,2:]

#%%
cluster_del_candidate = []
for num in cluster_num:
    num = int(num)
    ref_x = cluster_coords[num-1,0]
    ref_y = cluster_coords[num-1,1]
    ref_z = cluster_coords[num-1,2]
    for n in  range(num+1,len(cluster_coords) ):
        x = cluster_coords[n-1,0]
        y = cluster_coords[n-1,1]
        z = cluster_coords[n-1,2]

        if np.sqrt(( (ref_x-x)**2 + (ref_y-y)**2 + (ref_z-z)**2 ) ) < ref_distance:
            cluster_del_candidate.append(num)
            cluster_del_candidate.append(n)
        else:
            continue

        
cluster_virgo = np.setdiff1d(cluster_num,cluster_del_candidate)   

#%%

df1 = pd.DataFrame({'virgo': cluster_virgo})
df1.to_csv(ref_path  + 'virgo_info.csv')

#%%

        
        

# %%
