#%%
import copy
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import math
import shutil
import time
from numba import jit,njit, float32,complex64,int32

#%%


# %%

ref_path = '/storage/filament/works_v4/data/200Mpc_1/'

for cluster_num in np.sort(os.listdir(ref_path + 'clusters/')):
    print(cluster_num)

    
    filament_sig = np.load(ref_path + 'signature/pre/' + cluster_num)
    temp = np.load(ref_path + 'DL/raw_data/tem/' + cluster_num)

    volume_const  = 0.01*( np.float32(200/2048)*(3.086/0.7))**3
    xray = np.load(ref_path + 'DL/raw_data/xray/' + cluster_num ) + np.log10(volume_const)

    dens = np.load(ref_path + 'clusters/' + cluster_num)[:filament_sig.shape[0],:filament_sig.shape[0],:filament_sig.shape[0]]

    candidate_coordx = []
    candidate_coordy = []
    candidate_coordz = []
    vr = 10
    for iz in range(vr,filament_sig.shape[0]-vr):
        
        for iy in range(vr,filament_sig.shape[0]-vr):
            for ix in range(vr,filament_sig.shape[0]-vr):
                if xray[ix,iy,iz] >= -5.5:
                    

                    ref_value = xray[ix,iy,iz]
                    tmp_x = ix
                    tmp_y = iy
                    tmp_z = iz
                    max_value = ref_value
                    for rx in range(-vr,vr+1):
                        for ry in range(-vr,vr+1):
                            for rz in range(-vr,vr+1):
                                if ref_value < xray[ix+rx, iy+ry, iz+rz]:
                                    tmp_x = 0
                                    tmp_y = 0
                                    tmp_z = 0


                                else:
                                    pass

                    candidate_coordx.append(tmp_x)
                    candidate_coordy.append(tmp_y)
                    candidate_coordz.append(tmp_z)

    coords = list(set(list(zip(candidate_coordx,candidate_coordy,candidate_coordz))))
    coords = np.array(sorted(coords, key = lambda x:x[2]))[1:]
    
    labels = np.zeros([filament_sig.shape[0],filament_sig.shape[0],filament_sig.shape[0]])
    mean_signature = np.mean(filament_sig[filament_sig!=0])
    for ix in range(filament_sig.shape[0]):
        for iy in range(filament_sig.shape[0]):
            for iz in range(filament_sig.shape[0]):
                if temp[ix,iy,iz] <= 4:
                    labels[ix,iy,iz] = 0
                elif filament_sig[ix,iy,iz] >= mean_signature:
                    labels[ix,iy,iz] = 1


    for _,(ix,iy,iz) in enumerate(coords):
        if ix == 199 and iy == 199 and iz == 199:
            vr = 30
        else:
            vr = 10

        for i in range(ix-vr-1,ix+vr):
            for j in range(iy-vr-1,iy+vr):
                for k in range(iz-vr-1,iz+vr):

                    if np.sqrt((i-ix)**2 + (j-iy)**2 + (k-iz)**2) <= vr and temp[ix,iy,iz] > 4:
                        labels[i,j,k] = 1
                    else:
                        pass

    break

    # dens_binary = np.zeros([dens.shape[0],dens.shape[0],dens.shape[0]])
    # for i in range(dens.shape[0]):
    #     for j in range(dens.shape[0]):
    #         for k in range(dens.shape[0]):
    #             if temp[i,j,k] >= 4:
    #                 dens_binary[i,j,k] = 1
    #             else:
    #                 pass
#%%
    #labels = labels * dens_binary 
np.save(ref_path + 'DL/label/' + cluster_num + '_raw',labels)


#%%
data_set = np.zeros([cluster.shape[0]**3,4])

data_set[:,0] = cluster.flatten()
data_set[:,1] = filament.flatten()
data_set[:,2] = wall.flatten()
data_set[:,3] = dens.flatten()

start = time.time()
label_spread = LabelSpreading(kernel='knn', alpha=0.1,n_jobs=-1)
label_spread.fit(data_set[::64,:], labels.flatten()[::64])
end = time.time()

print("fit time",end-start)
# 655
#%%
start = time.time()
test = label_spread.predict(data_set)
end = time.time()
print("predict time",end-start)

#%%
from skimage.morphology import skeletonize
from skimage.morphology import medial_axis
#skeleton_lee = skeletonize(labels, method='lee')
skel, distance = medial_axis(labels, return_distance=True)
#%%

np.save(ref_path + 'DL/label/' + '9_skeleton.npy',skeleton_lee)
#%%

post_signature = np.load('/storage/filament/works_v4/data/200Mpc_1/signature/post/' + cluster_num)

binary = np.zeros([398,398,398])
for i in range(398):
    for j in range(398):
        for k in range(398):
            if temp[i,j,k] <= 4:
                binary[i,j,k] = 0 
            else:
                binary[i,j,k] = post_signature[i,j,k]

np.save(ref_path + 'DL/label/' + 'binary_post_temp.npy',binary)
# %%
