#%%
import numpy as np
import sys 
import os
import matplotlib.pyplot as plt
import copy 
import cv2


# %%
box_list = ['01','02']
subbox_list = ['1','2','3','4','5','6','7','8']

#subbox_num = ''

#subbox_list = ['1']

#box_list = ['02']
#subbox_list = ['2']

distance = '40Mpc'

if distance == '30Mpc':
    box_size = 37
if distance == '40Mpc':
    box_size = 50
for box_num in box_list:
    for subbox_num in subbox_list:
        data_path = '/storage/filament/data/200Mpc_v2/box'+box_num +'/'
        dens_result_path = '/storage/filament/result/cluster_3d/' + distance + '/density_temp/box' + box_num + '/subbox0' +subbox_num + '/dens/'
        xray_result_path = '/storage/filament/result/cluster_3d/' + distance + '/density_temp/box' + box_num + '/subbox0' +subbox_num + '/xray/'
        data_type = ['a','f','g']
        # a = density
        # f = temperature
        # g = xray   

        dens = np.zeros(256**3,dtype='float64')
        dens = np.genfromtxt(data_path + '256den18' + data_type[0] + '+1024+' + subbox_num,dtype='float64')
        dens = dens.reshape([256,256,256])

        temp = np.zeros(256**3,dtype='float64')
        temp = np.genfromtxt(data_path + '256den18' + data_type[1] + '+1024+' + subbox_num,dtype='float64')
        temp = temp.reshape([256,256,256])

        volume_const  = 0.01*( (200/2048)*(3.086/0.7))**3

        xray = np.zeros(256**3,dtype='float64')
        xray = np.genfromtxt(data_path + '256den18' + data_type[2] + '+1024+' + subbox_num,dtype='float64')
        xray = xray.reshape([256,256,256])
        xray = xray*volume_const

        index_path = '/storage/filament/data/200Mpc_v2/box'+box_num +'+Cinfo/'
        index = np.genfromtxt(index_path + 'box' + box_num + '+xlum+' + subbox_num+'.dat',dtype='uint8')

        x_index = index[:,3]-1
        y_index = index[:,2]-1
        z_index = index[:,1]-1

        coords = np.array(list(zip(x_index,y_index,z_index)))
        coords = np.array(sorted(coords, key=lambda x:x[2] ))

        box_list =[]
        dens_3d = np.zeros([box_size*2+1,box_size*2+1,box_size*2+1],dtype='float64')

        cluster_peak_thres = 2*1.16*10**7

        num=0

        print('box_num ', box_num, 'subbox_num ',subbox_num, 'density')

        for n,(ix,iy,iz) in enumerate(coords):
            if (ix < box_size or ix>256-box_size or iy < box_size or iy > 256-box_size or iz <box_size or iz >256-box_size):
                continue
            else:
                if temp[ix,iy,iz] >= cluster_peak_thres:
                    print(ix,iy,iz)
                    dens_3d[:,:,:] = dens[-box_size+ix:box_size+ix+1,-box_size+iy:box_size+iy+1,-box_size+iz:box_size+iz+1]               

                    if not os.path.isdir(dens_result_path + str(num) + '/'):
                         os.makedirs(dens_result_path + str(num) + '/')
                    os.chdir(dens_result_path + str(num) + '/')            

                    for nn in range(2*box_size+1):
                        plt.figure(figsize=[30,30])
                        plt.imsave(str(nn) +'.png',np.log10(dens_3d[:,:,nn]),vmin=0,vmax=4,cmap='gray',dpi=300,format='png')
 

                    candidate_coordx = []
                    candidate_coordy = []
                    candidate_coordz = []
                    log_xray = np.log10(xray)
                    
                    

                    for iiz in range(-box_size+iz,box_size+iz+1):
                        for iiy in range(-box_size+iy,box_size+iy+1):
                            for iix in range(-box_size+ix,box_size+ix+1):
                                if log_xray[iix,iiy,iiz] >= -5.5:
                                    
                                    radius = 0
                                    
                                    for vr in range(10):
                                        critical_xray = np.mean(log_xray[iix-vr:iix+vr , iiy-vr:iiy+vr , iiz-vr:iiz+vr])
                                        if critical_xray < -6:
                                            break
                                        else:
                                            pass


                                    # 주변 local maximun tag
                                    if vr >= 2 and vr <= 10:
                                        ref_value = log_xray[iix,iiy,iiz]
                                        tmp_x = iix-ix+box_size
                                        tmp_y = iiy-iy+box_size
                                        tmp_z = iiz-iz+box_size
                                        max_value = ref_value
                                        for rx in range(-vr,vr+1):
                                            for ry in range(-vr,vr+1):
                                                for rz in range(-vr,vr+1):
                                                    if ref_value < log_xray[iix+rx, iiy+ry, iiz+rz]:
                                                        tmp_x = 0
                                                        tmp_y = 0
                                                        tmp_z = 0
                                                        
                                                        
                                                    else:
                                                        pass
                                                    
                                    candidate_coordx.append(tmp_x)
                                    candidate_coordy.append(tmp_y)
                                    candidate_coordz.append(tmp_z)
                    

                    xray_coords = list(set(list(zip(candidate_coordx,candidate_coordy,candidate_coordz))))
                    xray_coords = np.array(sorted(xray_coords, key = lambda x:x[2]))      
                    xray_peaks = np.zeros([2*box_size+1,2*box_size+1,2*box_size+1])

                    for num_peak,(mx,my,mz) in enumerate(xray_coords):
                        
                        for xx in range(-4,5):
                            for yy in range(-4,5):
                                for zz in range(-4,5):
                                    try:
                                        if log_xray[mx+ix-box_size+xx,my+iy-box_size+yy,mz+iz-box_size+zz] > -7 and \
                                            mx+xx <= 100 and mx-xx >= 0 and my+yy <= 100 and my-yy >= 0 and mz+zz <= 100 and mz-zz >= 0:
                                            xray_peaks[mx+xx,my+yy,mz+zz] = num_peak
                                        else:
                                            pass
                                    except:
                                        pass


                    if not os.path.isdir(xray_result_path + str(num) + '/'):
                            os.makedirs(xray_result_path + str(num) + '/')
                    os.chdir(xray_result_path + str(num) + '/')            
            
                    np.savetxt('peak_coords',xray_coords,fmt='%3i')             

                    tmp_xray_coords = xray_peaks.reshape([101**3])
                    np.savetxt('xray_group',tmp_xray_coords,fmt='%3i')             

                    num = num+1

#%%


# %%
num=0
for n,(ix,iy,iz) in enumerate(coords):
    if (ix < box_size or ix>256-box_size or iy < box_size or iy > 256-box_size or iz <box_size or iz >256-box_size):
        continue
    else:
        if temp[ix,iy,iz] >= cluster_peak_thres:
            print(ix,iy,iz)
            dens_3d[:,:,:] = dens[-box_size+ix:box_size+ix+1,-box_size+iy:box_size+iy+1,-box_size+iz:box_size+iz+1]               

            if not os.path.isdir(dens_result_path + str(num) + '/'):
                    os.makedirs(dens_result_path + str(num) + '/')
            os.chdir(dens_result_path + str(num) + '/')            

            for nn in range(2*box_size+1):
                plt.figure(figsize=[30,30])
                plt.imsave(str(nn) +'.png',np.log10(dens_3d[:,:,nn]),vmin=0,vmax=4,cmap='gray',dpi=300,format='png')


            candidate_coordx = []
            candidate_coordy = []
            candidate_coordz = []
            log_xray = np.log10(xray)
            
            

            for iiz in range(-box_size+iz,box_size+iz+1):
                for iiy in range(-box_size+iy,box_size+iy+1):
                    for iix in range(-box_size+ix,box_size+ix+1):
                        if log_xray[iix,iiy,iiz] >= -5.5:
                            
                            radius = 0
                            
                            for vr in range(10):
                                critical_xray = np.mean(log_xray[iix-vr:iix+vr , iiy-vr:iiy+vr , iiz-vr:iiz+vr])
                                if critical_xray < -6:
                                    break
                                else:
                                    pass


                            # 주변 local maximun tag
                            if vr >= 2 and vr <= 10:
                                ref_value = log_xray[iix,iiy,iiz]
                                tmp_x = iix-ix+box_size
                                tmp_y = iiy-iy+box_size
                                tmp_z = iiz-iz+box_size
                                max_value = ref_value
                                for rx in range(-vr,vr+1):
                                    for ry in range(-vr,vr+1):
                                        for rz in range(-vr,vr+1):
                                            if ref_value < log_xray[iix+rx, iiy+ry, iiz+rz]:
                                                tmp_x = 0
                                                tmp_y = 0
                                                tmp_z = 0
                                                
                                                
                                            else:
                                                pass
                                            
                            candidate_coordx.append(tmp_x)
                            candidate_coordy.append(tmp_y)
                            candidate_coordz.append(tmp_z)
            

            xray_coords = list(set(list(zip(candidate_coordx,candidate_coordy,candidate_coordz))))
            xray_coords = np.array(sorted(xray_coords, key = lambda x:x[2]))      
            xray_peaks = np.zeros([2*box_size+1,2*box_size+1,2*box_size+1])

            for num_peak,(mx,my,mz) in enumerate(xray_coords):
                
                for xx in range(-4,5):
                    for yy in range(-4,5):
                        for zz in range(-4,5):
                            if log_xray[mx+ix-box_size+xx,my+iy-box_size+yy,mz+iz-box_size+zz] > -7 and \
                                mx+xx <= 100 and mx-xx > 0 and my+yy <= 100 and my-yy > 0 and mz+zz <= 100 and mz-zz > 0:
                                xray_peaks[mx+xx,my+yy,mz+zz] = num_peak
                            else:
                                pass
                    


            if not os.path.isdir(xray_result_path + str(num) + '/'):
                    os.makedirs(xray_result_path + str(num) + '/')
            os.chdir(xray_result_path + str(num) + '/')            
    
            np.savetxt('peak_coords',xray_coords,fmt='%3i')             

            tmp_xray_coords = xray_peaks.reshape([101**3])
            np.savetxt('xray_group',tmp_xray_coords,fmt='%3i')             

            num = num+1

# %%
i=20
plt.figure(figsize=[16,8])
plt.subplot(121)
plt.imshow(np.log10(dens_3d[i,:,:]))
plt.subplot(122)
plt.imshow(xray_peaks[i,:,:])
plt.gray()

# %%
