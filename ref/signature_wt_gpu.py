#%%
import numpy as np
import matplotlib.pyplot as plt
import copy 
from numpy.fft import rfftn,irfftn,irfft,ifft,fftn,ifftn
import time
from numba import jit,njit, float32,complex64,int32
from numba.np.linalg import eigvals_impl
import pandas as pd
import os 


@jit(float32(float32))
def step_function(input):
    if input >= 0:
        output = 1
    else:
        output = 0 

    return output


@jit(float32(float32))
def jacobian_matrix(input_array):
    length = input_array.shape[0]
    array_1 = np.zeros([3,length-1,length-1,length-1],dtype=np.float32)

    # gradient of function
    for i in range(length-1):
        for j in range(length-1):
            for k in range(length-1):

                array_1[0,i,j,k] = (input_array[i+1,j,k] - input_array[i,j,k])  # df/dx
                array_1[1,i,j,k] = (input_array[i,j+1,k] - input_array[i,j,k])  # df/dy
                array_1[2,i,j,k] = (input_array[i,j,k+1] - input_array[i,j,k])  # df/dz

    return array_1


@jit(float32(float32))
def hessian_matrix(input_array):
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



    for i in range(length-2):
        for j in range(length-2):
            for k in range(length-2):

                array_2[0,i,j,k] = (array_1[0,i+1,j,k] - array_1[0,i,j,k])
                array_2[1,i,j,k] = (array_1[1,i+1,j,k] - array_1[1,i,j,k])
                array_2[2,i,j,k] = (array_1[2,i+1,j,k] - array_1[2,i,j,k])

                array_2[3,i,j,k] = (array_1[0,i,j+1,k] - array_1[0,i,j,k])
                array_2[4,i,j,k] = (array_1[1,i,j+1,k] - array_1[1,i,j,k])
                array_2[5,i,j,k] = (array_1[2,i,j+1,k] - array_1[2,i,j,k])
                
                array_2[6,i,j,k] = (array_1[0,i,j,k+1] - array_1[0,i,j,k])
                array_2[7,i,j,k] = (array_1[1,i,j,k+1] - array_1[1,i,j,k])
                array_2[8,i,j,k] = (array_1[2,i,j,k+1] - array_1[2,i,j,k])

    return array_2


@jit(float32(float32,int32))
def smoothing_real_space(quantity,smoothing_scale):
    array_size = len(np.arange(0,cluster_grid,4))
    smoothing_array = np.zeros([array_size,array_size,array_size],dtype=np.float32)
    
    for ix,i in enumerate(np.arange(0,cluster_grid,4)):
        for iy,j in enumerate(np.arange(0,cluster_grid,4)):
            for iz,k in enumerate(np.arange(0,cluster_grid,4)):
                i = int(i)
                j = int(j)
                k = int(k)
           
                smoothing_array[ix,iy,iz] = np.mean(quantity[i:i+smoothing_scale,j:j+smoothing_scale,k:k+smoothing_scale])
    
    return smoothing_array

@jit(complex64(complex64,float32,int32,int32))
def smoothing_k_space(fft_array,smoothing_scale,cluster_length,cluster_grid):
    smoothing_array = copy.deepcopy(fft_array)
    cluster_length = 40
    # cluster_grid = 201

    for ix in range(int(fft_array.shape[0]/2)):
        for iy in range(int(fft_array.shape[1]/2)):
            for iz in range(int(fft_array.shape[2]/2)):

                k = np.sqrt( (ix/cluster_length)**2 + (iy/cluster_length)**2 + (iz/cluster_length)**2 )

                smoothing_array[ix,iy,iz] = fft_array[ix,iy,iz]*np.exp( (-(k**2)*smoothing_scale**2)/2)

        for iy in range(int(cluster_grid/2),cluster_grid-1):
            for iz in range(int(cluster_grid/2),cluster_grid-1):

                k = np.sqrt( (ix/cluster_length)**2 + ((iy-(cluster_grid-1))/cluster_length)**2 + ((iz-(cluster_grid-1))/cluster_length)**2  )

                smoothing_array[ix,iy,iz] = fft_array[ix,iy,iz]*np.exp( (-(k**2)*smoothing_scale**2)/2)

        for iy in range(int(cluster_grid/2),cluster_grid-1):
            for iz in range(int(fft_array.shape[2]/2)):

                k = np.sqrt( (ix/cluster_length)**2 + ((iy-(cluster_grid-1))/cluster_length)**2 + (iz/cluster_length)**2  )

                smoothing_array[ix,iy,iz] = fft_array[ix,iy,iz]*np.exp( (-(k**2)*smoothing_scale**2)/2)


        for iy in range(int(fft_array.shape[1]/2)):
            for iz in range(int(cluster_grid/2),cluster_grid-1):

                k = np.sqrt( (ix/cluster_length)**2 + ((iy)/cluster_length)**2 + ((iz-(cluster_grid-1))/cluster_length)**2  )

                smoothing_array[ix,iy,iz] = fft_array[ix,iy,iz]*np.exp( (-(k**2)*smoothing_scale**2)/2)
 

    for ix in range(int(cluster_grid/2),cluster_grid-1):
        for iy in range(int(cluster_grid/2),cluster_grid-1):
            for iz in range(int(cluster_grid/2),cluster_grid-1):

                k = np.sqrt( ((ix-(cluster_grid-1))/cluster_length)**2 + ((iy-(cluster_grid-1))/cluster_length)**2 + ((iz-(cluster_grid-1))/cluster_length)**2  )

                smoothing_array[ix,iy,iz] = fft_array[ix,iy,iz]*np.exp( (-(k**2)*smoothing_scale**2)/2)
                
        for iy in range(int(fft_array.shape[1]/2)):
            for iz in range(int(cluster_grid/2),cluster_grid-1):

                k = np.sqrt( ((ix-(cluster_grid-1))/cluster_length)**2 + ((iy)/cluster_length)**2 + ((iz-(cluster_grid-1))/cluster_length)**2  )

                smoothing_array[ix,iy,iz] = fft_array[ix,iy,iz]*np.exp( (-(k**2)*smoothing_scale**2)/2)

        for iy in range(int(cluster_grid/2),cluster_grid-1):
            for iz in range(int(fft_array.shape[1]/2)):

                k = np.sqrt( ((ix-(cluster_grid-1))/cluster_length)**2 + ((iy-(cluster_grid-1))/cluster_length)**2 + ((iz)/cluster_length)**2  )

                smoothing_array[ix,iy,iz] = fft_array[ix,iy,iz]*np.exp( (-(k**2)*smoothing_scale**2)/2)
                

        for iy in range(int(fft_array.shape[1]/2)):
            for iz in range(int(fft_array.shape[1]/2)):

                k = np.sqrt( ((ix-(cluster_grid-1))/cluster_length)**2 + ((iy)/cluster_length)**2 + ((iz)/cluster_length)**2  )

                smoothing_array[ix,iy,iz] = fft_array[ix,iy,iz]*np.exp( (-(k**2)*smoothing_scale**2)/2)        

    return smoothing_array    

#@jit(float32(float32))
@njit
def gpu_eig(h_mat):
    
    eigenval = np.zeros([3,h_mat.shape[-1],h_mat.shape[-1],h_mat.shape[-1]],dtype=np.float64)
    #eigenvec = np.zeros([3,h_mat.shape[-1],h_mat.shape[-1],h_mat.shape[-1]],dtype=np.float64)
    w = np.zeros(3,dtype=np.float32)
    tmp = h_mat.reshape([3,3,h_mat.shape[-1],h_mat.shape[-1],h_mat.shape[-1]])

    for i in range(h_mat.shape[-1]):
        for j in range(h_mat.shape[-1]):
            for k in range(h_mat.shape[-1]):

                w,eigenvec = np.linalg.eig(tmp[:,:,i,j,k])
                w = np.sort(w)
                if w[0] == 0.0:
                    w[0] = 1e-10
                eigenval[0,i,j,k] = w[0]  #lambda 1
                eigenval[1,i,j,k] = w[1]  #lambda 2
                eigenval[2,i,j,k] = w[2]  #lambda 3
    
    return eigenval,eigenvec

@jit(float32(float32))
def gpu_filament_signature(eigenvalues):
    
    filament_signature = np.zeros([eigenvalues.shape[-1],eigenvalues.shape[-1],eigenvalues.shape[-1]],dtype=np.float32)

    for i in range(eigenvalues.shape[-1]):
        for j in range(eigenvalues.shape[-1]):
            for k in range(eigenvalues.shape[-1]):

                filament_signature[i,j,k] = np.abs(eigenvalues[1,i,j,k]/eigenvalues[0,i,j,k]) * (1 - np.abs(eigenvalues[2,i,j,k]/eigenvalues[0,i,j,k])) * step_function(1 - np.abs(eigenvalues[2,i,j,k]/eigenvalues[0,i,j,k]))* \
                                np.abs(eigenvalues[1,i,j,k])*step_function(-eigenvalues[0,i,j,k])*step_function(-eigenvalues[1,i,j,k])

    return filament_signature


@jit(float32(float32))
def gpu_cluster_signature(eigenvalues):
    
    cluster_signature = np.zeros([eigenvalues.shape[-1],eigenvalues.shape[-1],eigenvalues.shape[-1]],dtype=np.float32)

    for i in range(eigenvalues.shape[-1]):
        for j in range(eigenvalues.shape[-1]):
            for k in range(eigenvalues.shape[-1]):

                cluster_signature[i,j,k] = np.abs(eigenvalues[2,i,j,k]/eigenvalues[0,i,j,k])* \
                                 np.abs(eigenvalues[2,i,j,k])*step_function(-eigenvalues[0,i,j,k])*step_function(-eigenvalues[1,i,j,k])*step_function(-eigenvalues[2,i,j,k])

    return cluster_signature


@jit(float32(float32))
def gpu_wall_signature(eigenvalues):
    
    wall_signature = np.zeros([eigenvalues.shape[-1],eigenvalues.shape[-1],eigenvalues.shape[-1]],dtype=np.float32)

    for i in range(eigenvalues.shape[-1]):
        for j in range(eigenvalues.shape[-1]):
            for k in range(eigenvalues.shape[-1]):
               
                wall_signature[i,j,k] = (1-np.abs(eigenvalues[1,i,j,k]/eigenvalues[0,i,j,k]))*step_function(1-np.abs(eigenvalues[1,i,j,k]/eigenvalues[0,i,j,k])) *\
                                (1-np.abs(eigenvalues[2,i,j,k]/eigenvalues[0,i,j,k]))*step_function(1-np.abs(eigenvalues[2,i,j,k]/eigenvalues[0,i,j,k])) *\
                                np.abs(eigenvalues[0,i,j,k])*step_function(-eigenvalues[0,i,j,k])


    return wall_signature



#%%
box_length = '300Mpc'
cluster_length = 45
box_list = ['1']

cluster_grid = 300   #45Mpc, Smoothing: 4, 0.586 Grid/Mpc   
smoothing_scale = 4


cluster_path = '/storage/filament/works_v6/300Mpc_1/cluster_box/xray/'
    
t = np.load(cluster_path + '1.npy')
jacobian_mat = jacobian_matrix(t)

np.save(cluster_path + 'test_jacobian',jacobian_mat)
# hessian_mat = hessian_matrix(t)
# eigenvalues,eigenvectors = gpu_eig(hessian_mat)
#%%

tmp = jacobian_mat[:,130:135,130:135,130:135].reshape([3,5,5,5])

for i in range(5):
    for j in range(5):
        for k in range(5):

            w,eigenvec = np.linalg.eig(tmp[:,i,j,k])
#     for cluster_num in np.sort(np.array(os.listdir(cluster_path))):

#         print(cluster_num)

#         t = np.load(cluster_path + cluster_num)

#         #dens_smoothing = smoothing_real_space(t,smoothing_scale)

#         eigenvalues = gpu_eig(hessian_mat)

#         filament_signature = gpu_filament_signature(eigenvalues)
#         wall_signature = gpu_wall_signature(eigenvalues)
#         cluster_signature = gpu_cluster_signature(eigenvalues)


#         if not os.path.isdir(save_path + 'filament/'):
#             os.makedirs(save_path + 'filament/')
#         if not os.path.isdir(save_path + 'wall/'):
#             os.makedirs(save_path + 'wall/')
#         if not os.path.isdir(save_path + 'cluster/'):
#             os.makedirs(save_path + 'cluster/')
#         if not os.path.isdir(save_path + 'dens/'):
#             os.makedirs(save_path + 'dens/')

#         np.save(save_path + 'filament/' +  str(cluster_num),filament_signature)    
#         np.save(save_path + 'wall/' + str(cluster_num),wall_signature)    
#         np.save(save_path + 'cluster/' + str(cluster_num),cluster_signature)    
#         #np.save(save_path + 'dens/' + str(cluster_num),dens_smoothing)   
        

# #%%
# smoothing_array = np.zeros([72,72,72],dtype=np.float32)

# for ix,i in enumerate(np.linspace(0,cluster_grid,72)[:-1]):
#     for iy,j in enumerate(np.linspace(0,cluster_grid,72)[:-1]):
#         for iz,k in enumerate(np.linspace(0,cluster_grid,72)[:-1]):
#             i = int(i)
#             j = int(j)
#             k = int(k)
        
#             smoothing_array[ix,iy,iz] = np.mean(t[i:i+4,j:j+4,k:k+4])