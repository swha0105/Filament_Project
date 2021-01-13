#%%
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import sys
import gc
#%%
ref_path = '/storage/filament/works_v7/300Mpc_1/'

threshold_list = []
for cluster_num in np.sort(np.array(os.listdir(ref_path + 'cluster_box/xray/'))):

    print(cluster_num)
    #cluster_num = '36.npy'
    label_ref = np.load(ref_path + 'label/post/' + cluster_num)
    xray_ref = np.load(ref_path + 'cluster_box/xray/' + cluster_num)

    label = np.zeros(xray_ref.shape)

    index_ref = (np.argwhere(label_ref == 1)) * 4


    for ix,iy,iz in index_ref:
        label[ix,iy,iz] = 1 


    label_post = copy.deepcopy(label)
    for ix,iy,iz in index_ref:
        
        index = np.argwhere(label[ix-4:ix+5,iy-4:iy+5,iz-4:iz+5] == 1)-4
        # index = np.argwhere(label[ix:ix+5,iy:iy+5,iz:iz+5] == 1)
        # print(index)

        for tmp_x,tmp_y,tmp_z in index:
            iix = tmp_x + ix
            iiy = tmp_y + iy
            iiz = tmp_z + iz

            min_x = min(iix,ix)
            min_y = min(iiy,iy)
            min_z = min(iiz,iz)

            max_x = max(iix,ix)
            max_y = max(iiy,iy)
            max_z = max(iiz,iz)

            label_post[min_x:max_x,min_y:max_y,min_z:max_z] = 1 


    log_mass_range = np.linspace(np.min(xray_ref),np.max(xray_ref),100)


    fraction_list = []
    for threshold in log_mass_range:
        
        volume_count = (xray_ref[xray_ref>threshold]).shape[0]

        fraction_list.append(volume_count / (xray_ref.shape[0]**3))
        #print(volume_count / (xray_ref.shape[0]**3),threshold)
        if volume_count / (xray_ref.shape[0]**3) < 0.1:
            threshold_list.append([int(cluster_num[:-4]),threshold])
            break


    label_post[xray_ref<threshold] = 0 

    np.save(ref_path + 'label/upsampling/' + cluster_num,label_post)
np.savetxt(ref_path + 'label/threshold_list',threshold_list)

# %%
