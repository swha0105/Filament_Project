#%%
import copy
import os
import sys
import matplotlib.pyplot as plt

import numpy as np
import gc
from numba import jit,njit, float32,int32
from scipy.ndimage import gaussian_filter

#%%
def subsampling(input):
    length = input.shape[0]
    output = np.zeros([int(length/2),int(length/2),int(length/2)],dtype=np.float32)

    for i in range(int(length/2)):
        for j in range(int(length/2)):
            for k in range(int(length/2)):
                output[i,j,k] = input[2*i,2*j,2*k]

    return output

def upsampling(input):
    length = input.shape[0]
    output = np.zeros([int(length*2),int(length*2),int(length*2)],dtype=np.float32)

    for i in range(int(length)):
        for j in range(int(length)):
            for k in range(int(length)):
                
                if i%2 == 0 or j%2 == 0 or k%2 == 0: 
                    output[2*i,2*j,2*k] = 0
                else:
                    output[2*i,2*j,2*k] = input[i,j,k]

    return output

# %%
box_length = '300Mpc'
cluster_length = 45
box_num = '1'
res = 2048

ref_path = '/storage/filament/works_v7/' + box_length + '_' + box_num + '/'
#virgo_list = np.loadtxt(ref_path + 'virgo_list')

for cluster_num in np.sort(np.array(os.listdir(ref_path + 'cluster_box/xray/'))):
    cluster_num = int(cluster_num[:-4])

    I1 = np.load(ref_path + 'cluster_box/xray/' + str(cluster_num) + '.npy')
    I2 = gaussian_filter(I1,sigma=1)
    I2 = subsampling(I2)
    I1_prime = upsampling(I2)
    I1_prime = gaussian_filter(I1_prime,sigma=1)
    L1 = I1-I1_prime

    np.save(ref_path + 'pyramid/xray/gaussian/1/' + str(cluster_num), I2)
    np.save(ref_path + 'pyramid/xray/laplacian/1/' + str(cluster_num), L1)

    I3 = gaussian_filter(I2,sigma=1)
    I3 = subsampling(I3)
    I2_prime = upsampling(I3)
    I2_prime = gaussian_filter(I2_prime,sigma=1)
    L2 = I2-I2_prime
    
    np.save(ref_path + 'pyramid/xray/gaussian/2/' + str(cluster_num), I3)
    np.save(ref_path + 'pyramid/xray/laplacian/2/' + str(cluster_num), L1)




# %%
box_length = '300Mpc'
cluster_length = 45
box_num = '1'
res = 2048

ref_path = '/storage/filament/works_v7/' + box_length + '_' + box_num + '/'
#virgo_list = np.loadtxt(ref_path + 'virgo_list')

for cluster_num in np.sort(np.array(os.listdir(ref_path + 'cluster_box/temp/'))):
    cluster_num = int(cluster_num[:-4])

    I1 = np.load(ref_path + 'cluster_box/temp/' + str(cluster_num) + '.npy')
    I2 = gaussian_filter(I1,sigma=1)
    I2 = subsampling(I2)
    I1_prime = upsampling(I2)
    I1_prime = gaussian_filter(I1_prime,sigma=1)
    L1 = I1-I1_prime

    np.save(ref_path + 'pyramid/temp/gaussian/1/' + str(cluster_num), I2)
    np.save(ref_path + 'pyramid/temp/laplacian/1/' + str(cluster_num), L1)

    I3 = gaussian_filter(I2,sigma=1)
    I3 = subsampling(I3)
    I2_prime = upsampling(I3)
    I2_prime = gaussian_filter(I2_prime,sigma=1)
    L2 = I2-I2_prime
    
    np.save(ref_path + 'pyramid/temp/gaussian/2/' + str(cluster_num), I3)
    np.save(ref_path + 'pyramid/temp/laplacian/2/' + str(cluster_num), L1)


