#%%
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import sys
import gc


ref_path = '/storage/filament/works_v7/300Mpc_1/'

#q_list = ['xray','temp','dens']
#q_list = ['temp','dens']
cluster_list = []
group_list = []

#cluster_grid = 273
cluster_grid = 300

res = 2048
halos_list = np.loadtxt(ref_path +  'data/' +  '300Mpc_clump_' + '01'  + '.dat')

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

virgo_tmp_list = np.setdiff1d(cluster_list,no_virgo_list)
virgo_list = []

for _,num in enumerate(virgo_tmp_list):
    if halos_list[num,6] < 3 and halos_list[num,6] > 2:
        virgo_list.append(num)

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



#%%
#q_list = ['xray','vx','vy','vz']
q_list = ['dens']

for q_name in q_list:
    q = np.fromfile(ref_path + 'data/L300' + q_name,dtype=np.float32).reshape([res,res,res])


    cluster_save_path = ref_path + 'cluster_box/' + q_name + '/'
    if not os.path.isdir(cluster_save_path):
        os.makedirs(cluster_save_path)
        
    for cluster_num,tmp_cluster in enumerate(virgo_info[:,:5]):
        n_c = int(tmp_cluster[0])
        ix = int(tmp_cluster[1])
        iy = int(tmp_cluster[2])
        iz = int(tmp_cluster[3])
        vr = np.around(tmp_cluster[4])
        
        cluster_q = []
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

                    if q_name =='xray' or q_name == 'temp':
                        cluster_q.append(np.log10(q[iix,iiy,iiz]))
                    else:
                        cluster_q.append((q[iix,iiy,iiz]))
                    

        cluster_q = np.array(cluster_q).reshape([cluster_grid,cluster_grid,cluster_grid])
        
        if not os.path.isdir(ref_path + 'cluster_box/' + q_name + '/'):
            os.makedirs(ref_path + 'cluster_box/' + q_name + '/')

        np.save(ref_path + 'cluster_box/' + q_name + '/' + str(cluster_num+1),cluster_q)
    
    del q
    gc.collect()
#%%
        
cluster_q = []
group_num = []
group_ix = []
group_iy = []
group_iz = []
group_vr = []
        
for cluster_num,tmp_cluster in enumerate(virgo_info[:,:5]):
    n_c = int(tmp_cluster[0])
    ix = int(tmp_cluster[1])
    iy = int(tmp_cluster[2])
    iz = int(tmp_cluster[3])
    vr = np.around(tmp_cluster[4])
    

    print(cluster_num)

    group_num.append(n_c)
    group_ix.append(ix)
    group_iy.append(iy)
    group_iz.append(iz)
    group_vr.append(vr)

    
info = list(zip(group_num,group_ix,group_iy,group_iz,group_vr))
np.savetxt(ref_path + 'cluster_box/virgo_info'  ,info)
