#%%
import copy
import os
import sys
import pandas as pd 
import cv2
import matplotlib.pyplot as plt
import numpy as np
#%%
#box_size_list = ['200Mpc','300Mpc','100Mpc']
#box_num_list = ['1','2']
box_size_list = ['300Mpc']
box_num_list = ['2']
resolution_list = ['1024']



for box_size in box_size_list:

    for  box_num in box_num_list:
        for resolution in resolution_list:

            data_path = '/storage/filament/works_v2/data_filament/' + box_size + '_' + box_num + '_' + resolution + '/'
            result_path = '/storage/filament/works_v2/result_filament/'  + box_size + '_' + box_num + '_' + resolution + '/'

            resolution = int(resolution)
            
            if box_size == '200Mpc':
                if resolution == 1024:
                    distance = 100
                elif resolution == 512:
                    distance = 50
            elif box_size == '300Mpc':
                if resolution == 1024:
                    distance = 70
                elif resolution == 512:
                    distance = 35
            elif box_size == '100Mpc':
                if resolution == 1024:
                    distance = 200
                elif resolution == 512:
                    distance = 100

            dens = np.fromfile(data_path + 'raw/' + 'box0' + box_num + 'den' + str(resolution) ,dtype='float32')
            dens = dens.reshape([resolution,resolution,resolution])

            volume_const  = 0.01*( (200/2048)*(3.086/0.7))**3        
            xray = np.fromfile(data_path + 'raw/' + 'box0' + box_num + 'xray' + str(resolution) ,dtype='float32')
            xray = xray.reshape([resolution,resolution,resolution])
            xray = xray*volume_const

            temp = np.fromfile(data_path + 'raw/' + 'box0' + box_num + 'temp' + str(resolution) ,dtype='float32')
            temp = temp.reshape([resolution,resolution,resolution])
            #%%
            cluster_center_x = []
            cluster_center_y = []
            cluster_center_z = []
            padd_resolution = resolution + 2*distance


            dens_padd = np.zeros([resolution+2*distance,resolution+2*distance,resolution+2*distance],dtype='float32')

            for i in range(resolution):
                for j in range(resolution):
                    for k in range(resolution):
                        dens_padd[distance+i,distance+j,distance+k] = dens[i,j,k]

            dens_padd[:distance,distance:resolution+distance,distance:resolution+distance] = dens[resolution-distance:resolution,:,:]
            dens_padd[resolution + distance : padd_resolution,distance:resolution+distance,distance:resolution+distance] = dens[:distance,:,:]

            dens_padd[:padd_resolution,:distance,:padd_resolution] = dens_padd[:padd_resolution,resolution:resolution+distance,:padd_resolution]
            dens_padd[:padd_resolution,resolution+distance:padd_resolution,:padd_resolution] = dens_padd[:padd_resolution,distance:2*distance ,:padd_resolution]

            dens_padd[:padd_resolution,:padd_resolution,:distance,] = dens_padd[:padd_resolution,:padd_resolution,resolution:resolution+distance,]
            dens_padd[:padd_resolution,:padd_resolution,resolution+distance:padd_resolution] = dens_padd[:padd_resolution,:padd_resolution,distance:2*distance ]


            #%%
            candidate_coordx = []
            candidate_coordy = []
            candidate_coordz = []
            virial_radius_list = []
            virial_mass = []

            virial_max = int(4/((int(box_size[:3])/resolution)) )


            for iz in range(virial_max,resolution-virial_max):
                for iy in range(virial_max,resolution-virial_max):
                    for ix in range(virial_max,resolution-virial_max):
                        if temp[ix,iy,iz] >= 2*1.16*10**7 and xray[ix,iy,iz] >= 10**(-5.5):
                            print(iz,iy,ix)
                            virial_radius = 0
                            # virial raius 계산
                            for vr in range(virial_max):
                                critical_density = np.mean(dens[ix-vr:ix+vr+1 , iy-vr:iy+vr+1, iz-vr:iz+vr+1])
                                if (critical_density < 200):
                                    virial_radius = vr
                                    virial_density = critical_density
                                    break;
                                else:
                                    pass
                            
                            # 주변 local maximun tag
                            if not virial_radius ==0:
                                ref_value = xray[ix,iy,iz]
                                tmp_x = ix
                                tmp_y = iy
                                tmp_z = iz
                                max_value = ref_value
                                dens_sum = 0 
                                for rx in range(-virial_radius,virial_radius+1):
                                    for ry in range(-virial_radius,virial_radius+1):
                                        for rz in range(-virial_radius,virial_radius+1):

                                            if ref_value < xray[ix+rx, iy+ry, iz+rz]:
                                                tmp_x = 0
                                                tmp_y = 0
                                                tmp_z = 0
                                                
                                            else:
                                                pass
                    
                                virial_radius_list.append(virial_radius)                                
                                candidate_coordx.append(tmp_x)
                                candidate_coordy.append(tmp_y)
                                candidate_coordz.append(tmp_z)



                                log_m = np.log10(virial_density) + np.log10(0.044) + np.log10(1.879) + np.log10((0.7)**2) - 29
                                log_length = 3*(np.log10(virial_radius) + np.log10(0.39) - np.log10(0.7) + np.log10(3.086) + 24)
                                virial_mass.append(log_m + log_length)
                        else:
                            continue

            quantity = list(set(list(zip(candidate_coordx,candidate_coordy,candidate_coordz,virial_radius_list,virial_mass))))


            x_coords = []
            y_coords = []
            z_coords = []

            mass_list = []
            virial_list = []
            cluster_num = [] 

            cluster_count = 0 
            for n,(ix,iy,iz,vr,mass) in enumerate(quantity):
                if ix!=0 and iy!=0 and iz!=0:
                    cluster_count = cluster_count + 1
                    print(cluster_count)
                    cluster_num.append(cluster_count)
                    x_coords.append(int(ix))
                    y_coords.append(int(iy))
                    z_coords.append(int(iz))
                    virial_list.append(vr)
                    mass_list.append(mass - 33 - np.log10(2)  )
                else:
                    continue



            refined_quantity = np.array(list(set(list(zip(x_coords,y_coords,z_coords,virial_list,mass_list)))))

            df1 = pd.DataFrame({'cluster_num': cluster_num, 'x': refined_quantity[:,0], 'y': refined_quantity[:,1], 'z': refined_quantity[:,2],\
                'virial_grid':refined_quantity[:,3],'mass (Solar Mass)': refined_quantity[:,4] })

            
            save_path = '/storage/filament/works_v2/result_filament/'  + box_size + '_' + box_num + '_' + str(resolution) + '_info/'
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            
            df1.to_csv('/storage/filament/works_v2/result_filament/'  + box_size + '_' + box_num + '_' + str(resolution) + '_info/' + 'clusters_info.csv')

            cluster_count = 0
            cluster_num = [] 
            import matplotlib
            matplotlib.use('Agg')
            for nn,(ix,iy,iz,_,_) in enumerate(refined_quantity):
                ix = int(ix)
                iy = int(iy)
                iz = int(iz)
                cluster_count = cluster_count + 1

                cluster_num.append(cluster_count)

                dens_3d = np.zeros([2*distance+1,2*distance+1,2*distance+1],dtype='float32')
                dens_3d = dens_padd[ix:ix+2*distance+1,iy:iy+2*distance+1,iz:iz+2*distance+1]

                ##
                raw_data_path = data_path + 'clusters/' + str(cluster_count) + '/'
                if not os.path.isdir(raw_data_path):
                    os.makedirs(raw_data_path)

                np.save(raw_data_path + 'dens',dens_3d.reshape([(2*distance+1)**3]))

                #
                dens_path = result_path + str(cluster_count) +'/dens_img/'
                if not os.path.isdir(dens_path):
                    os.makedirs(dens_path)

                for nn in range(2*distance+1):
                    plt.figure(figsize=[30,30])
                    plt.imsave(dens_path + str(nn),np.log10(dens_3d[nn,:,:]),vmin=0,vmax=4,cmap='gray',dpi=300,format='png')

    

#%%



#%%

