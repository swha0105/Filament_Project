#%%
import copy
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import gc

box_length = '300Mpc'
cluster_length = 45
box_num = '1'
res = 2048

if box_length == '200Mpc':  
    cluster_grid = 450   #45Mpc, Smoothing: 6, 0.586 Grid/Mpc  

if box_length == '300Mpc':
    cluster_grid = 300   #45Mpc, Smoothing: 4, 0.586 Grid/Mpc   

if box_length == '100Mpc':
    cluster_grid = 900   #45Mpc, Smootinhg: 12, 0.586 Grid/Mpc   

ref_path = '/storage/filament/works_v5/' + box_length + '_' + box_num + '/clusters/'

group_list = np.loadtxt(ref_path+'group_info')
virgo_list = np.loadtxt(ref_path+'virgo_list')

#%%
for cluster_num in os.listdir(ref_path + 'pyramid/gaussian/2/'):
    data = np.load(ref_path + 'pyramid/gaussian/2/' + cluster_num )
    label = np.full((data.shape[0],data.shape[0],data.shape[0]),0)
    log_mass_range = np.linspace(np.min(data),np.max(data),100)
    vr = np.loadtxt(ref_path +'cluster_info/' + cluster_num[:-4])[4]
    #/storage/filament/works_v5/300Mpc_1/clusters/cluster_info
    
    
    fraction_list = []
    for threshold in log_mass_range:
        
        volume_count = (data[data>threshold]).shape[0]
        fraction_list.append(volume_count / (data.shape[0]**3))

        if volume_count / (data.shape[0]**3) < 0.1:
            print(cluster_num,threshold)
            break

    label[data>threshold] = -1


    for ix in range(int(label.shape[0]/2)-int(vr),int(label.shape[0]/2)+int(vr)+1):
        for iy in range(int(label.shape[0]/2)-int(vr),int(label.shape[0]/2)+int(vr)+1):
            for iz in range(int(label.shape[0]/2)-int(vr),int(label.shape[0]/2)+int(vr)+1):
                if np.sqrt( (ix-int(label.shape[0]/2))**2 + (iy-int(label.shape[0]/2))**2 + (iz-int(label.shape[0]/2))**2) <= 4:
                    label[ix,iy,iz] = 1
    
    np.save(ref_path + 'label/raw/' + cluster_num, label) 


#%%
for cluster_num in os.listdir(ref_path + 'pyramid/gaussian/2/'):
    data = np.load(ref_path + 'pyramid/gaussian/2/' + cluster_num )
    label = np.full((data.shape[0],data.shape[0],data.shape[0]),-1)
    log_mass_range = np.linspace(np.min(data),np.max(data),100)

    fraction_list = []
    for threshold in log_mass_range:
        
        volume_count = (data[data>threshold]).shape[0]
        fraction_list.append(volume_count / (data.shape[0]**3))

        if volume_count / (data.shape[0]**3) < 0.1:
            print(threshold)
            break

    label[data>threshold] = 0

    #np.save(ref_path + 'virgo/label_tmp/' + cluster_num, label) 

#for cluster_num in virgo_list:

    number_of_group = []
    group_vr_list = []
    group_xray_list = []
    group_temp_list = []
    group_dens_list = []

    #cluster_num = str(int(cluster_num))

    label = np.load(ref_path + 'virgo/label_tmp/' + cluster_num )

    cluster_info = np.loadtxt(ref_path + 'cluster_info/'+ cluster_num[:-4])

    cluster_ix = int(cluster_info[1])
    cluster_iy = int(cluster_info[2])
    cluster_iz = int(cluster_info[3])
    cluster_vr = int(cluster_info[4])

    tmp_ix = cluster_ix
    tmp_iy = cluster_iy
    tmp_iz = cluster_iz

    if cluster_ix < int(cluster_grid/2):
        tmp_ix = res + cluster_ix
    if cluster_iy < int(cluster_grid/2):
        tmp_iy = res + cluster_iy
    if cluster_iz < int(cluster_grid/2):
        tmp_iz = res + cluster_iz

    for _,tmp in enumerate(group_list):
        group_ix = int(tmp[1])
        group_iy = int(tmp[2])
        group_iz = int(tmp[3])
        group_vr = int(np.around((tmp[4])*0.15/0.6))
        group_dens = tmp[5]
        group_temp = tmp[6]
        group_xray = tmp[7]

        tmp_iix = group_ix
        tmp_iiy = group_iy
        tmp_iiz = group_iz

        if group_ix < int(cluster_grid/2):
            tmp_iix = res + group_ix
        if group_iy < int(cluster_grid/2):
            tmp_iiy = res + group_iy
        if group_iz < int(cluster_grid/2):
            tmp_iiz = res + group_iz

        if (np.abs(cluster_ix - group_ix) < int(cluster_grid/2) and np.abs(cluster_iy - group_iy) < int(cluster_grid/2) and np.abs(cluster_iz - group_iz) < int(cluster_grid/2)):
            dens_sum_tmp = []

            new_x = int(np.around((group_ix - cluster_ix)/4)) + int(label.shape[0]/2)
            new_y = int(np.around((group_iy - cluster_iy)/4)) + int(label.shape[0]/2)
            new_z = int(np.around((group_iz - cluster_iz)/4)) + int(label.shape[0]/2)

            for ix in range(new_x-group_vr,new_x+group_vr+1):
                for iy in range(new_y-group_vr,new_y+group_vr+1):
                    for iz in range(new_z-group_vr,new_z+group_vr+1):
                        
                        if np.sqrt( (ix-new_x)**2 + (iy-new_y)**2 + (iz-new_z)**2) <= group_vr:
                            try:
                                tmp_sum = data[ix,iy,iz]
                            except:
                                pass
                            dens_sum_tmp.append(tmp_sum)
                            
            #print(dens_sum_tmp)
            #if np.mean(dens_sum_tmp) > threshold:
            for ix in range(new_x-group_vr,new_x+group_vr+1):
                for iy in range(new_y-group_vr,new_y+group_vr+1):
                    for iz in range(new_z-group_vr,new_z+group_vr+1):
                    
                        if np.sqrt( (ix-new_x)**2 + (iy-new_y)**2 + (iz-new_z)**2) <= group_vr:
                            try:
                                label[ix,iy,iz] = 1
                            except:
                                pass

            
                number_of_group.append('group')
                group_xray_list.append(group_xray)
                group_temp_list.append(group_temp)
                group_dens_list.append(group_dens)
                group_vr_list.append(group_vr)
            
        elif (np.abs(tmp_ix-tmp_iix) < int(cluster_grid/2) and np.abs(tmp_iy-tmp_iiy) < int(cluster_grid/2) and np.abs(tmp_iz-tmp_iiz) < int(cluster_grid/2)):
            dens_sum_tmp = []

            new_x = int(np.around((tmp_iix - tmp_ix)/4)) + int(label.shape[0]/2)
            new_y = int(np.around((tmp_iiy - tmp_iy)/4)) + int(label.shape[0]/2)
            new_z = int(np.around((tmp_iiz - tmp_iz)/4)) + int(label.shape[0]/2)

            for ix in range(new_x-group_vr,new_x+group_vr+1):
                for iy in range(new_y-group_vr,new_y+group_vr+1):
                    for iz in range(new_z-group_vr,new_z+group_vr+1):
                        
                        if np.sqrt( (ix-new_x)**2 + (iy-new_y)**2 + (iz-new_z)**2) <= group_vr:
                            
                            try:
                                tmp_sum = data[ix,iy,iz]
                            except:
                                pass
                            dens_sum_tmp.append(tmp_sum)
                            
            #print(dens_sum_tmp)
            #if np.mean(dens_sum_tmp) > threshold:
            for ix in range(new_x-group_vr,new_x+group_vr+1):
                for iy in range(new_y-group_vr,new_y+group_vr+1):
                    for iz in range(new_z-group_vr,new_z+group_vr+1):
                    
                        if np.sqrt( (ix-new_x)**2 + (iy-new_y)**2 + (iz-new_z)**2) <= group_vr:
                            try:
                                label[ix,iy,iz] = 1
                            except:
                                pass

            
                number_of_group.append('group')
                group_xray_list.append(group_xray)
                group_temp_list.append(group_temp)
                group_dens_list.append(group_dens)
                group_vr_list.append(group_vr)
 
    np.save(ref_path + 'virgo/label/' + cluster_num, label) 
    print(cluster_num)
    break

#%%
#data = np.load(ref_path + 'pyramid/gaussian/2/' + cluster_num + '.npy')
i=22
plt.figure(figsize=[16,8])
plt.subplot(121)
plt.imshow(label[:,:,i])
plt.colorbar()
plt.subplot(122)
plt.imshow(data[:,:,i])
# %%
print(cluster_ix,cluster_iy,cluster_iz)
print(group_ix,group_iy,group_iz)
#%%

#73,16,5

print(data[73,15,5])