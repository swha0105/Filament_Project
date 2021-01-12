#%%
import numpy as np
import sys 
import os
import matplotlib.pyplot as plt
os.chdir('/storage/filament/codes/')
from utils import *
import copy 

#%%
temp = np.genfromtxt('/storage/filament/data/raw/256den18f+1024',dtype='float64')
xray = np.genfromtxt('/storage/filament/data/raw/256den18g+1024',dtype='float64')

temp_mat = temp.reshape([256,256,256])
xray_mat = xray.reshape([256,256,256])
index = np.genfromtxt('/storage/filament/data/raw/box01+xlum.dat',dtype='uint8')
#%%
index = np.genfromtxt('/storage/filament/data/raw/box01+xlum.dat',dtype='uint8')

x_index = index[:,3]-1
y_index = index[:,2]-1
z_index = index[:,1]-1

box_size = 37

#temp_3d = np.zeros([box_size*2,box_size*2,box_size*2,len(coords)],dtype='float64')
temp_3d = np.zeros([box_size*2,box_size*2,box_size*2],dtype='float64')
coords = np.array(list(zip(x_index,y_index,z_index)))
box_list =[]
#%%
n=0
temp_3d = np.zeros([box_size*2,box_size*2,box_size*2],dtype='float64')
for _,(ix,iy,iz) in enumerate(coords):
    if (ix < box_size or ix>256-box_size or iy < box_size or iy > 256-box_size or iz <box_size or iz >256-box_size):
        continue
    else:
        print(ix,iy,iz)
        temp_3d[:,:,:] = temp_mat[-box_size+ix:box_size+ix,-box_size+iy:box_size+iy,-box_size+iz:box_size+iz]
       
        box_list.append(temp_mat[-box_size+ix:box_size+ix,-box_size+iy:box_size+iy,-box_size+iz:box_size+iz])


box_list = np.array(box_list)

# n=0
# for _,(ix,iy,iz) in enumerate(coords):
#     if (ix < box_size or ix>256-box_size or iy < box_size or iy > 256-box_size or iz <box_size or iz >256-box_size):
#         continue
#     else:
#         temp_3d[:,:,:,n] = temp_mat[-box_size+ix:box_size+ix,-box_size+iy:box_size+iy,-box_size+iz:box_size+iz]
#         n = n+1
        

#%%
temp_1d = temp_3d.reshape( (box_size*2)**3,10)
np.savetxt('3d_temp',temp_1d)
# %%
i = 3

plt.contourf(np.log10(box_list[10,:,:,37]))

# %%
