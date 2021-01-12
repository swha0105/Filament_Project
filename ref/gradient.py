#%%
import numpy as np
import matplotlib.pyplot as plt
import copy 
from numpy.fft import rfftn,irfftn,irfft,ifft,fftn,ifftn
import time
from numba import jit,njit, float32,complex64,int32
import pandas as pd
import os 


@jit(float32(float32))
def gradient(input_array):
    length = input_array.shape[0]
    array_1 = np.zeros([3,length-1,length-1,length-1],dtype=np.float32)
    array_2 = np.zeros([9,length-2,length-2,length-2],dtype=np.float32)

    # gradient of function
    for i in range(length-1):
        for j in range(length-1):
            for k in range(length-1):

                array_1[0,i,j,k] = (input_array[i+1,j,k] - input_array[i,j,k])  # df/dx
                array_1[1,i,j,k] = (input_array[i,j+1,k] - input_array[i,j,k])  # df/dy
                array_1[2,i,j,k] = (input_array[i,j,k+1] - input_array[i,j,k])  # df/dz

    return array_1
# %%
box_length = '300Mpc'
cluster_length = 45
box_list = ['1']

if box_length == '200Mpc':  
    cluster_grid = 450   #45Mpc, Smoothing: 6, 0.586 Grid/Mpc  
    smoothing_scale = 6
if box_length == '300Mpc':
    cluster_grid = 300   #45Mpc, Smoothing: 4, 0.586 Grid/Mpc   
    smoothing_scale = 4
if box_length == '100Mpc':
    cluster_grid = 900   #45Mpc, Smootinhg: 12, 0.586 Grid/Mpc   
    smoothing_scale = 12

for box_num in box_list:

    ref_path = '/storage/filament/works_v6/' + box_length + '_' + box_num + '/'
    cluster_path = ref_path +   'cluster_box/xray/'
    save_path = ref_path + 'gradient/'

    for cluster_num in np.sort(np.array(os.listdir(cluster_path))):

        print(cluster_num)

        
        t = np.load(cluster_path + cluster_num)

        gradient_data = gradient(t)

        if not os.path.isdir(save_path):
            os.makedirs(save_path )
        
        np.save(save_path +  str(cluster_num),gradient_data)    
        