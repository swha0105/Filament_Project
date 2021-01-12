#%%
import copy
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import gc
from numba import jit,njit, float32,int32
from scipy.ndimage import gaussian_filter
# %%
box_length = '300Mpc'
cluster_length = 45
box_list = ['1']
res = 2048

if box_length == '200Mpc':  
    cluster_grid = 450   #45Mpc, Smoothing: 6, 0.586 Grid/Mpc  

if box_length == '300Mpc':
    cluster_grid = 300   #45Mpc, Smoothing: 4, 0.586 Grid/Mpc   

if box_length == '100Mpc':
    cluster_grid = 900   #45Mpc, Smootinhg: 12, 0.586 Grid/Mpc   

q_list = ['dens','xray','temp']
cluster_list = []
group_list = []


for box_num in box_list:

    for q_name in q_list:
    ref_path = '/storage/filament/works_v6/' + box_length + '_' + box_num + '/'
    q = np.fromfile(ref_path + 'data/L300' + q_name,dtype=np.float32).reshape([res,res,res])
    halos_list = np.loadtxt(ref_path +  'data/' + box_length + '_clump_' + '0' + box_num + '.dat')


    for halo_num in range(len(halos_list)):
        
        if halos_list[halo_num,7] >= 1 and halos_list[halo_num,6] >= 2:
            cluster_list.append(halo_num)
        elif (halos_list[halo_num,7] >= 0.01 and halos_list[halo_num,7] < 1) and ( halos_list[halo_num,6] >= 0.3 and halos_list[halo_num,6] < 2) :
            group_list.append(halo_num)

cluster_list = np.array(cluster_list)
group_list = np.array(group_list)

#%%
no_virgo_list = []

for n in cluster_list:
        
    ix = int(halos_list[n,1])
    iy = int(halos_list[n,2])
    iz = int(halos_list[n,3])

    tmp_ix = ix
    tmp_iy = iy
    tmp_iz = iz

    if ix < int(cluster_grid/2):
        tmp_ix = res + ix
    if iy < int(cluster_grid/2):
        tmp_iy = res + iy
    if iz < int(cluster_grid/2):
        tmp_iz = res + iz
    
    
    for nn in cluster_list:

        iix = int(halos_list[nn,1])
        iiy = int(halos_list[nn,2])
        iiz = int(halos_list[nn,3])

        if ix == iix and iy == iiy and iz == iiz:
            continue
        else:

            tmp_iix = iix
            tmp_iiy = iiy
            tmp_iiz = iiz

            if iix < int(cluster_grid/2):
                tmp_iix = res + iix
            if iiy < int(cluster_grid/2):
                tmp_iiy = res + iiy
            if iiz < int(cluster_grid/2):
                tmp_iiz = res + iiz
        

            if (np.abs(ix-iix) < int(cluster_grid/2) and np.abs(iy-iiy) < int(cluster_grid/2) and np.abs(iz-iiz) < int(cluster_grid/2)) or \
                (np.abs(tmp_ix-tmp_iix) < int(cluster_grid/2) and np.abs(tmp_iy-tmp_iiy) < int(cluster_grid/2) and np.abs(tmp_iz-tmp_iiz) < int(cluster_grid/2)):

                no_virgo_list.append(n)


no_virgo_list = np.array(list(set(no_virgo_list)))

virgo_list = np.setdiff1d(cluster_list,no_virgo_list)
virgo_info = np.zeros([len(virgo_list),8])

# halo_num,ix,iy,iz, vr(grid), dens(solar mass), temp (xray weighted, keV), xray lum (10^44)
for n,num in enumerate(virgo_list):
    virgo_info[n,0] = num
    virgo_info[n,1] = int(halos_list[num,3])
    virgo_info[n,2] = int(halos_list[num,2])
    virgo_info[n,3] = int(halos_list[num,1])
    virgo_info[n,4] = halos_list[num,4]
    virgo_info[n,5] = halos_list[num,5]
    virgo_info[n,6] = halos_list[num,6]
    virgo_info[n,7] = halos_list[num,7]


# excluded virial_radius == 0.6
group_list_wo_no_vr = []
group_info = np.zeros([len(group_list),8])
n=0
for num in group_list:
    
    if halos_list[num,4] == 4.1:
        group_list_wo_no_vr.append(num)


group_list_wt_vr = np.setdiff1d(group_list,group_list_wo_no_vr)
group_info = np.zeros([len(group_list_wt_vr),8])

for n,num in enumerate(group_list_wt_vr):
    group_info[n,0] = num
    group_info[n,1] = int(halos_list[num,3])
    group_info[n,2] = int(halos_list[num,2])
    group_info[n,3] = int(halos_list[num,1])
    group_info[n,4] = halos_list[num,4]
    group_info[n,5] = halos_list[num,5]
    group_info[n,6] = halos_list[num,6]
    group_info[n,7] = halos_list[num,7]




cluster_save_path = ref_path + 'clusters/' 
for cluster_num,tmp_cluster in enumerate(virgo_info[:,:5]):
    n_c = int(tmp_cluster[0])
    ix = int(tmp_cluster[1])
    iy = int(tmp_cluster[2])
    iz = int(tmp_cluster[3])
    vr = int(np.around(tmp_cluster[4]*0.15/0.6) )
    
    cluster_dens = []
    group_num = []
    group_ix = []
    group_iy = []
    group_iz = []
    group_vr = []
    
    print(cluster_num)

    group_num.append(n_c)
    group_ix.append(ix)
    group_iy.append(iy)
    group_iz.append(iz)
    group_vr.append(vr)

    for iiz in range(iz - int(cluster_grid/2), iz + int(cluster_grid/2)):
        for iiy in range(iy - int(cluster_grid/2), iy + int(cluster_grid/2)):
            for iix in range(ix - int(cluster_grid/2), ix + int(cluster_grid/2)):
                
                if iiz < 0:
                    iiz = iiz+res
                elif iiz >= res:
                    iiz = iiz-res
                else:
                    iiz = iiz

                if iiy < 0:
                    iiy = iiy+res
                elif iiy >= res:
                    iiy = iiy-res
                else:
                    iiy = iiy

                if iix < 0:
                    iix = iix+res
                elif iix >= res:
                    iix = iix-res
                else:
                    iix = iix

                
                for n,tmp in enumerate(group_info[:,:5]):
                    g_c = int(tmp[0])
                    g_ix = int(tmp[1])
                    g_iy = int(tmp[2])
                    g_iz = int(tmp[3])
                    g_vr = tmp[3]

                    if g_ix == iix and g_iy == iiy and g_iz == iiz:
                        group_num.append(g_c)
                        group_ix.append(g_ix)
                        group_iy.append(g_iy)
                        group_iz.append(g_iz)
                        group_vr.append(g_vr)




                cluster_dens.append(np.log10(dens[iix,iiy,iiz]))

    cluster_dens = np.array(cluster_dens).reshape([cluster_grid,cluster_grid,cluster_grid])
    np.save(cluster_save_path + 'dens/' + str(cluster_num+1),cluster_dens)

    group_array = np.array(list(zip(group_num,group_ix,group_iy,group_iz,group_vr)))
    np.savetxt(cluster_save_path + 'cluster_info/' + str(cluster_num+1),group_array)
    
    np.savetxt(cluster_save_path + 'group_info',group_info)


#%%
virgo_num_list = []

for i in os.listdir(cluster_save_path + 'cluster_info/'):
    a = np.loadtxt(cluster_save_path + 'cluster_info/' + i )

    if halos_list[int(a[0]),6] >= 2.0 and halos_list[int(a[0]),6] <= 3.0:
        virgo_num_list.append(int(i))
    
np.savetxt(cluster_save_path  + 'virgo_list',virgo_num_list,fmt='%i')

# %%
