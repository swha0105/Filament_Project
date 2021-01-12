#%%
import numpy as np
import matplotlib.pyplot as plt
import copy 
import time
import pandas as pd
import os 

# %%
virgo_info = np.loadtxt('/storage/filament/works_v6/300Mpc_1/cluster_box/virgo_info')

ref_path = '/storage/filament/works_v6/300Mpc_1/data/'
halo_info = np.loadtxt(ref_path + '300Mpc_clump_01.dat')

cluster_grid = 270
#%%

for cluster_num,tmp_cluster in enumerate(virgo_info):

    halo_list = []
    halo_list_ix = []
    halo_list_iy = []
    halo_list_iz = []
    halo_list_vr = []
    virgo_num = tmp_cluster[0]
    virgo_ix = tmp_cluster[3]
    virgo_iy = tmp_cluster[2]
    virgo_iz = tmp_cluster[1]
    virgo_vr = tmp_cluster[4]


    halo_list.append(virgo_num)
    halo_list_ix.append(virgo_ix)
    halo_list_iy.append(virgo_iy)
    halo_list_iz.append(virgo_iz)
    halo_list_vr.append(virgo_vr)    
    for halo_num,tmp_halo in enumerate(halo_info):
        halo_num = tmp_halo[0]
        halo_ix = tmp_halo[3]
        halo_iy = tmp_halo[2]
        halo_iz = tmp_halo[1]
        halo_vr = tmp_halo[4]        
        halo_dens = tmp_halo[5]
        halo_temp = tmp_halo[6]
        halo_xray = tmp_halo[7]
        

        if (halo_xray >= 0.01 and halo_xray < 1) and (halo_temp >= 0.3 and halo_temp < 2) and np.abs(virgo_ix-halo_ix) < int(cluster_grid/2) \
             and np.abs(virgo_iy-halo_iy) < int(cluster_grid/2) and  np.abs(virgo_iz-halo_iz) < int(cluster_grid/2) and halo_vr != 4.1:

            halo_list.append(halo_num)
            halo_list_ix.append(halo_ix)
            halo_list_iy.append(halo_iy)
            halo_list_iz.append(halo_iz)
            halo_list_vr.append(halo_vr)

    save_info = list(zip(halo_list,halo_list_ix,halo_list_iy,halo_list_iz,halo_list_vr))
    np.savetxt(ref_path + 'group_info/' + str(cluster_num+1),save_info  )
    
# %%
for cluster_num,tmp_cluster in enumerate(virgo_info):
    print(tmp_cluster.shape)

# %%
