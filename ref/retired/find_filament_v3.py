#%%
import copy
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

#%%

def find_candidate_coords(previous_x,previous_y,previous_z,filament_x,filament_y,filament_z,num):
    test_box = filament_conneted[previous_x-1:previous_x+2,previous_y-1:previous_y+2,previous_z-1:previous_z+2]
    candidate_set = np.array(np.where(test_box==1)).T

    ## candidate_set coords convert into real coords       
    for j in range(candidate_set.shape[0]):
        candidate_set[j][0] = candidate_set[j][0] - 1 + previous_x
        candidate_set[j][1] = candidate_set[j][1] - 1 + previous_y
        candidate_set[j][2] = candidate_set[j][2] - 1 + previous_z

    x_candidate = []
    y_candidate = []
    z_candidate = []
    
    filament_array = np.array(list(zip(filament_x[str(num)],filament_y[str(num)],filament_z[str(num)])))


    for m in range(candidate_set.shape[0]):
        candidate_zip = np.array([candidate_set[m][0],candidate_set[m][1],candidate_set[m][2]])
        #print("candidate",candidate_zip,"filament",filament_array)
        
        if (candidate_zip == filament_array).all(1).any():
        #if candidate_zip[:] not in filament_array[n][:]:
            continue            
        else:

            x_candidate.append(candidate_set[m][0])
            y_candidate.append(candidate_set[m][1])
            z_candidate.append(candidate_set[m][2])
         #   print("filament_append")            


    candidate_filament = np.array(list(zip(x_candidate,y_candidate,z_candidate)))
    return candidate_filament



#%%

box_list = ['01','02']
subbox_list = ['1','2','3','4','5','6','7','8']

for box_name in box_list:
    for subbox_name in subbox_list:
                
        
        dens_path = '/storage/filament/result/cluster_3d/40Mpc/density_temp/box' + box_name + '/subbox0' +  subbox_name + '/dens/'
        peak_list = np.sort(os.listdir(dens_path))
        for peak_name in peak_list:

            path = '/storage/filament/result/cluster_3d/40Mpc/density_temp/box' + box_name + '/subbox0' +  subbox_name 
            save_path = '/storage/filament/result/cluster_3d/40Mpc/density_temp/box' + box_name + '/subbox0' + subbox_name + '/filament/' + peak_name + '/'                           
            data_type = ['a','f','g']
                    
            filament_conneted = np.fromfile(path + '/label/bw/' + str(peak_name) ,dtype='int32')
            filament_conneted = filament_conneted.reshape([101,101,101])

            filament_major = np.fromfile(path + '/label/major/' + str(peak_name) ,dtype='int32')
            filament_major[filament_major==1] = 2
            filament_major = filament_major.reshape([101,101,101])


            dens = np.zeros([101,101,101])

            for nn,_ in enumerate(os.listdir(path + '/dens/' + str(peak_name) + '/')):
                dens[nn,:,:] = np.rot90(cv2.imread(path + '/dens/' + str(peak_name) + '/' + str(nn) + '.png',cv2.IMREAD_GRAYSCALE),3)

            box_size = 8 
            center_point = 50

            coords_x = []
            coords_y = []
            coords_z = []

            # x = 42
            plane = filament_conneted[center_point - box_size, center_point-box_size : center_point+box_size+1  ,center_point-box_size : center_point+box_size+1 ]
            coords = np.array(np.where(plane==1))

            for n in range(coords.shape[1]):
                coords_1 = center_point + coords[0,n] - box_size
                coords_2 = center_point + coords[1,n] - box_size 

                coords_x.append(center_point-box_size)
                coords_y.append(coords_1)
                coords_z.append(coords_2)

            # x = 58 
            plane = filament_conneted[center_point + box_size, center_point-box_size : center_point+box_size+1  ,center_point-box_size : center_point+box_size+1 ]
            coords = np.array(np.where(plane==1))

            for n in range(coords.shape[1]):
                coords_1 = center_point + coords[0,n] - box_size
                coords_2 = center_point + coords[1,n] - box_size 

                coords_x.append(center_point+box_size)
                coords_y.append(coords_1)
                coords_z.append(coords_2)

            # y = 42
            plane = filament_conneted[center_point-box_size : center_point+box_size+1, center_point - box_size  ,center_point-box_size : center_point+box_size+1 ]
            coords = np.array(np.where(plane==1))

            for n in range(coords.shape[1]):
                coords_1 = center_point + coords[0,n] - box_size
                coords_2 = center_point + coords[1,n] - box_size 

                coords_x.append(coords_1)
                coords_y.append(center_point-box_size)
                coords_z.append(coords_2)    

            # y = 58
            plane = filament_conneted[center_point-box_size : center_point+box_size+1, center_point + box_size  ,center_point-box_size : center_point+box_size+1 ]
            coords = np.array(np.where(plane==1))

            for n in range(coords.shape[1]):
                coords_1 = center_point + coords[0,n] - box_size
                coords_2 = center_point + coords[1,n] - box_size 

                coords_x.append(coords_1)
                coords_y.append(center_point+box_size)
                coords_z.append(coords_2)       

            # z = 42
            plane = filament_conneted[center_point-box_size : center_point+box_size+1, center_point-box_size : center_point+box_size+1 , center_point - box_size  ]
            coords = np.array(np.where(plane==1))

            for n in range(coords.shape[1]):
                coords_1 = center_point + coords[0,n] - box_size
                coords_2 = center_point + coords[1,n] - box_size 

                coords_x.append(coords_1)
                coords_y.append(coords_2)
                coords_z.append(center_point-box_size)    

            # z = 58
            plane = filament_conneted[center_point-box_size : center_point+box_size+1 ,center_point-box_size : center_point+box_size+1 ,center_point + box_size]
            coords = np.array(np.where(plane==1))

            for n in range(coords.shape[1]):
                coords_1 = center_point + coords[0,n] - box_size
                coords_2 = center_point + coords[1,n] - box_size 

                coords_x.append(coords_1)
                coords_y.append(coords_2)
                coords_z.append(center_point+box_size)   
                
            coords = np.array(list(zip(coords_x,coords_y,coords_z)))

            filament_conneted[center_point - box_size+1 : center_point + box_size,center_point - box_size+1 : center_point + box_size,center_point - box_size+1 : center_point + box_size] = 0



            filament = {}    
            num = 0

            filament_x = {}
            filament_y = {}
            filament_z = {}


            for n in range(coords.shape[0]):

                ref_x = coords[n,0]
                ref_y = coords[n,1]
                ref_z = coords[n,2]

                previous_x = ref_x
                previous_y = ref_y
                previous_z = ref_z

                filament_x[str(num)] = [previous_x] 
                filament_y[str(num)] = [previous_y]
                filament_z[str(num)] = [previous_z]

                while True:
                    candidate_coords_filament = find_candidate_coords(previous_x,previous_y,previous_z,filament_x,filament_y,filament_z,num)
                    
                    if candidate_coords_filament.shape[0] == 0:

                        filament_zip = list(zip(filament_x[str(num)],filament_y[str(num)],filament_z[str(num)]))
                        filament[str(num)] = filament_zip
                        num = num + 1
                    
                        break


                    elif candidate_coords_filament.shape[0] == 1:

                        previous_x = candidate_coords_filament[0,0]
                        previous_y = candidate_coords_filament[0,1]
                        previous_z = candidate_coords_filament[0,2]

                        filament_x[str(num)].append(previous_x)
                        filament_y[str(num)].append(previous_y)
                        filament_z[str(num)].append(previous_z)


                    elif candidate_coords_filament.shape[0] >= 2:
                        

                        filament_brunch_x = {}
                        filament_brunch_y = {}
                        filament_brunch_z = {}
            
                        dens_mean = np.zeros(candidate_coords_filament.shape[0])

                        for num_filament_candidate,(previous_x,previous_y,previous_z) in enumerate(candidate_coords_filament):
                            major_true = 0

                            if filament_major[previous_x,previous_y,previous_z] == 2:
                                filament_x[str(num)].append(previous_x)
                                filament_y[str(num)].append(previous_y)
                                filament_z[str(num)].append(previous_z)

                                major_true = 1
                                break

                            filament_brunch_x[str(num_filament_candidate)] = copy.deepcopy(filament_x[str(num)])
                            filament_brunch_y[str(num_filament_candidate)] = copy.deepcopy(filament_y[str(num)])
                            filament_brunch_z[str(num_filament_candidate)] = copy.deepcopy(filament_z[str(num)])
                            
                            filament_brunch_x[str(num_filament_candidate)].append(previous_x)
                            filament_brunch_y[str(num_filament_candidate)].append(previous_y)
                            filament_brunch_z[str(num_filament_candidate)].append(previous_z)

                            while True:
                                        
                                candidate_coords_filament_brunch = find_candidate_coords(previous_x,previous_y,previous_z, \
                                filament_brunch_x,filament_brunch_y,filament_brunch_z,num_filament_candidate)

                                if candidate_coords_filament_brunch.shape[0] == 1:
                                    
                                    previous_x = candidate_coords_filament_brunch[0,0]
                                    previous_y = candidate_coords_filament_brunch[0,1]
                                    previous_z = candidate_coords_filament_brunch[0,2]
                                
                                    filament_brunch_x[str(num_filament_candidate)].append(previous_x)
                                    filament_brunch_y[str(num_filament_candidate)].append(previous_y)
                                    filament_brunch_z[str(num_filament_candidate)].append(previous_z)
                                
                                else:
                                    break


                        # density 비교
                        if major_true == 0:
                            for brunch_num in range(candidate_coords_filament.shape[0]):
                                x_coords = filament_brunch_x[str(brunch_num)]
                                y_coords = filament_brunch_y[str(brunch_num)]
                                z_coords = filament_brunch_z[str(brunch_num)]


                                dens_sum = 0
                                for nn in range(len(x_coords)):
                                    dens_sum = dens_sum + dens[x_coords[nn],y_coords[nn],z_coords[nn]]

                                dens_mean[brunch_num]  = dens_sum/len(x_coords)


                            brunch_mean_max = np.argmax(dens_mean)

                            filament_x[str(num)] = filament_brunch_x[str(brunch_mean_max)]
                            filament_y[str(num)] = filament_brunch_y[str(brunch_mean_max)]
                            filament_z[str(num)] = filament_brunch_z[str(brunch_mean_max)]

                            previous_x = filament_x[str(num)][-1]
                            previous_y = filament_y[str(num)][-1]
                            previous_z = filament_z[str(num)][-1]


            for i in list(filament.keys()):
                tmp_filament = filament[str(i)]
           
                np.savetxt(save_path + str(i),tmp_filament,fmt='%3i')



# In[16]:
### Add box

box_list = ['01_add','02_add']

for box_name in box_list:
    ref_path = '/storage/filament/result/cluster_3d/40Mpc/density_temp/box' + box_name + '/'

    for subbox_name in os.listdir(ref_path):

        path = ref_path + subbox_name + '/'                    
        dens_path = path + 'dens/'
        
        save_path = path + 'filament/'
        
                
        filament_conneted = np.fromfile(path + '/label/bw/0'  ,dtype='int32')
        filament_conneted = filament_conneted.reshape([101,101,101])

        filament_major = np.fromfile(path + '/label/major/0' ,dtype='int32')
        filament_major[filament_major==1] = 2
        filament_major = filament_major.reshape([101,101,101])


        dens = np.zeros([101,101,101])

        for nn,_ in enumerate(os.listdir(dens_path )):
            dens[nn,:,:] = np.rot90(cv2.imread(dens_path + str(nn) + '.png',cv2.IMREAD_GRAYSCALE),3)

        box_size = 8 
        center_point = 50

        coords_x = []
        coords_y = []
        coords_z = []

        # x = 42
        plane = filament_conneted[center_point - box_size, center_point-box_size : center_point+box_size+1  ,center_point-box_size : center_point+box_size+1 ]
        coords = np.array(np.where(plane==1))

        for n in range(coords.shape[1]):
            coords_1 = center_point + coords[0,n] - box_size
            coords_2 = center_point + coords[1,n] - box_size 

            coords_x.append(center_point-box_size)
            coords_y.append(coords_1)
            coords_z.append(coords_2)

        # x = 58 
        plane = filament_conneted[center_point + box_size, center_point-box_size : center_point+box_size+1  ,center_point-box_size : center_point+box_size+1 ]
        coords = np.array(np.where(plane==1))

        for n in range(coords.shape[1]):
            coords_1 = center_point + coords[0,n] - box_size
            coords_2 = center_point + coords[1,n] - box_size 

            coords_x.append(center_point+box_size)
            coords_y.append(coords_1)
            coords_z.append(coords_2)

        # y = 42
        plane = filament_conneted[center_point-box_size : center_point+box_size+1, center_point - box_size  ,center_point-box_size : center_point+box_size+1 ]
        coords = np.array(np.where(plane==1))

        for n in range(coords.shape[1]):
            coords_1 = center_point + coords[0,n] - box_size
            coords_2 = center_point + coords[1,n] - box_size 

            coords_x.append(coords_1)
            coords_y.append(center_point-box_size)
            coords_z.append(coords_2)    

        # y = 58
        plane = filament_conneted[center_point-box_size : center_point+box_size+1, center_point + box_size  ,center_point-box_size : center_point+box_size+1 ]
        coords = np.array(np.where(plane==1))

        for n in range(coords.shape[1]):
            coords_1 = center_point + coords[0,n] - box_size
            coords_2 = center_point + coords[1,n] - box_size 

            coords_x.append(coords_1)
            coords_y.append(center_point+box_size)
            coords_z.append(coords_2)       

        # z = 42
        plane = filament_conneted[center_point-box_size : center_point+box_size+1, center_point-box_size : center_point+box_size+1 , center_point - box_size  ]
        coords = np.array(np.where(plane==1))

        for n in range(coords.shape[1]):
            coords_1 = center_point + coords[0,n] - box_size
            coords_2 = center_point + coords[1,n] - box_size 

            coords_x.append(coords_1)
            coords_y.append(coords_2)
            coords_z.append(center_point-box_size)    

        # z = 58
        plane = filament_conneted[center_point-box_size : center_point+box_size+1 ,center_point-box_size : center_point+box_size+1 ,center_point + box_size]
        coords = np.array(np.where(plane==1))

        for n in range(coords.shape[1]):
            coords_1 = center_point + coords[0,n] - box_size
            coords_2 = center_point + coords[1,n] - box_size 

            coords_x.append(coords_1)
            coords_y.append(coords_2)
            coords_z.append(center_point+box_size)   
            
        coords = np.array(list(zip(coords_x,coords_y,coords_z)))

        filament_conneted[center_point - box_size+1 : center_point + box_size,center_point - box_size+1 : center_point + box_size,center_point - box_size+1 : center_point + box_size] = 0



        filament = {}    
        num = 0

        filament_x = {}
        filament_y = {}
        filament_z = {}


        for n in range(coords.shape[0]):

            ref_x = coords[n,0]
            ref_y = coords[n,1]
            ref_z = coords[n,2]

            previous_x = ref_x
            previous_y = ref_y
            previous_z = ref_z

            filament_x[str(num)] = [previous_x] 
            filament_y[str(num)] = [previous_y]
            filament_z[str(num)] = [previous_z]

            while True:
                candidate_coords_filament = find_candidate_coords(previous_x,previous_y,previous_z,filament_x,filament_y,filament_z,num)
                
                if candidate_coords_filament.shape[0] == 0:

                    filament_zip = list(zip(filament_x[str(num)],filament_y[str(num)],filament_z[str(num)]))
                    filament[str(num)] = filament_zip
                    num = num + 1
                
                    break


                elif candidate_coords_filament.shape[0] == 1:

                    previous_x = candidate_coords_filament[0,0]
                    previous_y = candidate_coords_filament[0,1]
                    previous_z = candidate_coords_filament[0,2]

                    filament_x[str(num)].append(previous_x)
                    filament_y[str(num)].append(previous_y)
                    filament_z[str(num)].append(previous_z)


                elif candidate_coords_filament.shape[0] >= 2:
                    

                    filament_brunch_x = {}
                    filament_brunch_y = {}
                    filament_brunch_z = {}
        
                    dens_mean = np.zeros(candidate_coords_filament.shape[0])

                    for num_filament_candidate,(previous_x,previous_y,previous_z) in enumerate(candidate_coords_filament):
                        major_true = 0

                        if filament_major[previous_x,previous_y,previous_z] == 2:
                            filament_x[str(num)].append(previous_x)
                            filament_y[str(num)].append(previous_y)
                            filament_z[str(num)].append(previous_z)

                            major_true = 1
                            break

                        filament_brunch_x[str(num_filament_candidate)] = copy.deepcopy(filament_x[str(num)])
                        filament_brunch_y[str(num_filament_candidate)] = copy.deepcopy(filament_y[str(num)])
                        filament_brunch_z[str(num_filament_candidate)] = copy.deepcopy(filament_z[str(num)])
                        
                        filament_brunch_x[str(num_filament_candidate)].append(previous_x)
                        filament_brunch_y[str(num_filament_candidate)].append(previous_y)
                        filament_brunch_z[str(num_filament_candidate)].append(previous_z)

                        while True:
                                    
                            candidate_coords_filament_brunch = find_candidate_coords(previous_x,previous_y,previous_z, \
                            filament_brunch_x,filament_brunch_y,filament_brunch_z,num_filament_candidate)

                            if candidate_coords_filament_brunch.shape[0] == 1:
                                
                                previous_x = candidate_coords_filament_brunch[0,0]
                                previous_y = candidate_coords_filament_brunch[0,1]
                                previous_z = candidate_coords_filament_brunch[0,2]
                            
                                filament_brunch_x[str(num_filament_candidate)].append(previous_x)
                                filament_brunch_y[str(num_filament_candidate)].append(previous_y)
                                filament_brunch_z[str(num_filament_candidate)].append(previous_z)
                            
                            else:
                                break


                    # density 비교
                    if major_true == 0:
                        for brunch_num in range(candidate_coords_filament.shape[0]):
                            x_coords = filament_brunch_x[str(brunch_num)]
                            y_coords = filament_brunch_y[str(brunch_num)]
                            z_coords = filament_brunch_z[str(brunch_num)]


                            dens_sum = 0
                            for nn in range(len(x_coords)):
                                dens_sum = dens_sum + dens[x_coords[nn],y_coords[nn],z_coords[nn]]

                            dens_mean[brunch_num]  = dens_sum/len(x_coords)


                        brunch_mean_max = np.argmax(dens_mean)

                        filament_x[str(num)] = filament_brunch_x[str(brunch_mean_max)]
                        filament_y[str(num)] = filament_brunch_y[str(brunch_mean_max)]
                        filament_z[str(num)] = filament_brunch_z[str(brunch_mean_max)]

                        previous_x = filament_x[str(num)][-1]
                        previous_y = filament_y[str(num)][-1]
                        previous_z = filament_z[str(num)][-1]


        for i in list(filament.keys()):
            tmp_filament = filament[str(i)]
        
            np.savetxt(save_path + str(i),tmp_filament,fmt='%3i')




# %%
