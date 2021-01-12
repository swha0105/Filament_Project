#%%
import copy
import os
import sys
import shutil
import cv2 
import matplotlib.pyplot as plt
import numpy as np
import copy

#%%

def find_candidate_coords(previous_x,previous_y,previous_z,filament_x,filament_y,filament_z,num):
    test_box = skeleton[previous_x-1:previous_x+2,previous_y-1:previous_y+2,previous_z-1:previous_z+2]
    candidate_set = np.array(np.where(test_box==1)).T

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
        
        if (candidate_zip == filament_array).all(1).any():
            continue            
        else:

            x_candidate.append(candidate_set[m][0])
            y_candidate.append(candidate_set[m][1])
            z_candidate.append(candidate_set[m][2])


    candidate_filament = np.array(list(zip(x_candidate,y_candidate,z_candidate)))
    return candidate_filament



#%%
    
skeleton_path = '/storage/filament/works_v5/300Mpc_1/clusters/label/skeleton/'


skeleton = np.load(skeleton_path + '1.npy')
#skeleton = skeleton[:72,:72,:72]
filament_end_point = np.array(np.where(skeleton==2)).T

filament_start_point = [int(skeleton.shape[0]/2),int(skeleton.shape[0]/2),int(skeleton.shape[0]/2)]

filament = {}    
num = 0

coords_x = []
coords_y = []
coords_z = []

#%%
filament = {}    
num = 0

filament_x = {}
filament_y = {}
filament_z = {}


for ref_x,ref_y,ref_z in filament_end_point:


    previous_x = ref_x
    previous_y = ref_y
    previous_z = ref_z

    filament_x[str(num)] = [previous_x] 
    filament_y[str(num)] = [previous_y]
    filament_z[str(num)] = [previous_z]

    while True:
        candidate_coords_filament = find_candidate_coords(previous_x,previous_y,previous_z,filament_x,filament_y,filament_z,num)
        
        if (candidate_coords_filament == [36,36,36]).all():
            break

        elif candidate_coords_filament.shape[0] == 1:

            previous_x = candidate_coords_filament[0,0]
            previous_y = candidate_coords_filament[0,1]
            previous_z = candidate_coords_filament[0,2]

            filament_x[str(num)].append(previous_x)
            filament_y[str(num)].append(previous_y)
            filament_z[str(num)].append(previous_z)


        elif candidate_coords_filament.shape[0] >= 2:
            

            filament_branch_x = {}
            filament_branch_y = {}
            filament_branch_z = {}

            for num_filament_candidate,(previous_x,previous_y,previous_z) in enumerate(candidate_coords_filament):

                filament_branch_x[str(num_filament_candidate)] = copy.deepcopy(filament_x[str(num)])
                filament_branch_y[str(num_filament_candidate)] = copy.deepcopy(filament_y[str(num)])
                filament_branch_z[str(num_filament_candidate)] = copy.deepcopy(filament_z[str(num)])
                
                filament_branch_x[str(num_filament_candidate)].append(previous_x)
                filament_branch_y[str(num_filament_candidate)].append(previous_y)
                filament_branch_z[str(num_filament_candidate)].append(previous_z)

                while True:
                            
                    candidate_coords_filament_branch = find_candidate_coords(previous_x,previous_y,previous_z, \
                    filament_branch_x,filament_branch_y,filament_branch_z,num_filament_candidate)

                    if candidate_coords_filament_branch.shape[0] == 1:
                        
                        previous_x = candidate_coords_filament_branch[0,0]
                        previous_y = candidate_coords_filament_branch[0,1]
                        previous_z = candidate_coords_filament_branch[0,2]
                    
                        filament_branch_x[str(num_filament_candidate)].append(previous_x)
                        filament_branch_y[str(num_filament_candidate)].append(previous_y)
                        filament_branch_z[str(num_filament_candidate)].append(previous_z)
                    
                    else:
                        break
            
        
            for branch_num in range(candidate_coords_filament.shape[0]):
                x_coords = filament_branch_x[str(branch_num)]
                y_coords = filament_branch_y[str(branch_num)]
                z_coords = filament_branch_z[str(branch_num)]


                # dens_sum = 0
                # for nn in range(len(x_coords)):
                #     dens_sum = dens_sum + dens[x_coords[nn],y_coords[nn],z_coords[nn]]

                # dens_mean[branch_num]  = dens_sum
                

            break
            branch_mean_max = np.argmax(dens_mean)

            filament_x[str(num)] = copy.deepcopy(filament_branch_x[str(branch_mean_max)])
            filament_y[str(num)] = copy.deepcopy(filament_branch_y[str(branch_mean_max)])
            filament_z[str(num)] = copy.deepcopy(filament_branch_z[str(branch_mean_max)])

            previous_x = filament_x[str(num)][-1]
            previous_y = filament_y[str(num)][-1]
            previous_z = filament_z[str(num)][-1]
            
            break

    filament_zip = list(zip(filament_x[str(num)],filament_y[str(num)],filament_z[str(num)]))
    filament[str(num)] = filament_zip
    num = num + 1

#%%
box_length = '300Mpc'
box_num = '1'

if box_length == '200Mpc':
    cluster_grid = 200*2 + 1
if box_length == '300Mpc':
    cluster_grid = 135*2 + 1
    
data_path = '/storage/filament/works_v4/data/' + box_length + '_' + box_num  + '/clusters/'
save_path = '/storage/filament/works_v4/data/' + box_length + '_' + box_num  + '/filament/segmented/'
skeleton_path = '/storage/filament/works_v4/data/' + box_length + '_' + box_num  + '/filament/skeletonized/'

box_size = int(cluster_grid/12)  #3Mpc
center_point = int(cluster_grid/2)
mpc_grid = int(box_length[:3])/2048

dens = np.zeros([cluster_grid-3,cluster_grid-3,cluster_grid-3])
filament_major = np.zeros([cluster_grid-3,cluster_grid-3,cluster_grid-3])
#tmp_cluster = ['125.npy']
for cluster_num in np.array(sorted(os.listdir(data_path))):
#for cluster_num in tmp_cluster:

    print(cluster_num)
    cluster_num = str(cluster_num)

    dens_path = data_path + cluster_num
    dens = np.load(dens_path).reshape([cluster_grid,cluster_grid,cluster_grid])

    filament_connected = np.load(skeleton_path + cluster_num)

    coords_x = []
    coords_y = []
    coords_z = []
   
    plane = filament_connected[center_point - box_size, center_point-box_size : center_point+box_size+1  ,center_point-box_size : center_point+box_size+1 ]
    coords = np.array(np.where(plane==1))


    for n in range(coords.shape[1]):
        coords_1 = center_point + coords[0,n] - box_size
        coords_2 = center_point + coords[1,n] - box_size 

        coords_x.append(center_point-box_size)
        coords_y.append(coords_1)
        coords_z.append(coords_2)

    plane = filament_connected[center_point + box_size, center_point-box_size : center_point+box_size+1  ,center_point-box_size : center_point+box_size+1 ]
    coords = np.array(np.where(plane==1))

    for n in range(coords.shape[1]):
        coords_1 = center_point + coords[0,n] - box_size
        coords_2 = center_point + coords[1,n] - box_size 

        coords_x.append(center_point+box_size)
        coords_y.append(coords_1)
        coords_z.append(coords_2)

    
    plane = filament_connected[center_point-box_size : center_point+box_size+1, center_point - box_size  ,center_point-box_size : center_point+box_size+1 ]
    coords = np.array(np.where(plane==1))

    for n in range(coords.shape[1]):
        coords_1 = center_point + coords[0,n] - box_size
        coords_2 = center_point + coords[1,n] - box_size 

        coords_x.append(coords_1)
        coords_y.append(center_point-box_size)
        coords_z.append(coords_2)    

    
    plane = filament_connected[center_point-box_size : center_point+box_size+1, center_point + box_size  ,center_point-box_size : center_point+box_size+1 ]
    coords = np.array(np.where(plane==1))

    for n in range(coords.shape[1]):
        coords_1 = center_point + coords[0,n] - box_size
        coords_2 = center_point + coords[1,n] - box_size 

        coords_x.append(coords_1)
        coords_y.append(center_point+box_size)
        coords_z.append(coords_2)       

    
    plane = filament_connected[center_point-box_size : center_point+box_size+1, center_point-box_size : center_point+box_size+1 , center_point - box_size  ]
    coords = np.array(np.where(plane==1))

    for n in range(coords.shape[1]):
        coords_1 = center_point + coords[0,n] - box_size
        coords_2 = center_point + coords[1,n] - box_size 

        coords_x.append(coords_1)
        coords_y.append(coords_2)
        coords_z.append(center_point-box_size)    

    
    plane = filament_connected[center_point-box_size : center_point+box_size+1 ,center_point-box_size : center_point+box_size+1 ,center_point + box_size]
    coords = np.array(np.where(plane==1))

    for n in range(coords.shape[1]):
        coords_1 = center_point + coords[0,n] - box_size
        coords_2 = center_point + coords[1,n] - box_size 

        coords_x.append(coords_1)
        coords_y.append(coords_2)
        coords_z.append(center_point+box_size)   
        
    coords = np.array(list(zip(coords_x,coords_y,coords_z)))

    filament_connected[center_point - box_size+1 : center_point + box_size,center_point - box_size+1 : center_point + box_size,center_point - box_size+1 : center_point + box_size] = 0

    
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
                break

            elif candidate_coords_filament.shape[0] == 1:

                previous_x = candidate_coords_filament[0,0]
                previous_y = candidate_coords_filament[0,1]
                previous_z = candidate_coords_filament[0,2]

                filament_x[str(num)].append(previous_x)
                filament_y[str(num)].append(previous_y)
                filament_z[str(num)].append(previous_z)


            elif candidate_coords_filament.shape[0] >= 2:
                

                filament_branch_x = {}
                filament_branch_y = {}
                filament_branch_z = {}
    
                dens_mean = np.zeros(candidate_coords_filament.shape[0])

                for num_filament_candidate,(previous_x,previous_y,previous_z) in enumerate(candidate_coords_filament):

                    filament_branch_x[str(num_filament_candidate)] = copy.deepcopy(filament_x[str(num)])
                    filament_branch_y[str(num_filament_candidate)] = copy.deepcopy(filament_y[str(num)])
                    filament_branch_z[str(num_filament_candidate)] = copy.deepcopy(filament_z[str(num)])
                    
                    filament_branch_x[str(num_filament_candidate)].append(previous_x)
                    filament_branch_y[str(num_filament_candidate)].append(previous_y)
                    filament_branch_z[str(num_filament_candidate)].append(previous_z)

                    while True:
                                
                        candidate_coords_filament_branch = find_candidate_coords(previous_x,previous_y,previous_z, \
                        filament_branch_x,filament_branch_y,filament_branch_z,num_filament_candidate)

                        if candidate_coords_filament_branch.shape[0] == 1:
                            
                            previous_x = candidate_coords_filament_branch[0,0]
                            previous_y = candidate_coords_filament_branch[0,1]
                            previous_z = candidate_coords_filament_branch[0,2]
                        
                            filament_branch_x[str(num_filament_candidate)].append(previous_x)
                            filament_branch_y[str(num_filament_candidate)].append(previous_y)
                            filament_branch_z[str(num_filament_candidate)].append(previous_z)
                        
                        else:
                            break
               
            
                for branch_num in range(candidate_coords_filament.shape[0]):
                    x_coords = filament_branch_x[str(branch_num)]
                    y_coords = filament_branch_y[str(branch_num)]
                    z_coords = filament_branch_z[str(branch_num)]


                    dens_sum = 0
                    for nn in range(len(x_coords)):
                        dens_sum = dens_sum + dens[x_coords[nn],y_coords[nn],z_coords[nn]]

                    dens_mean[branch_num]  = dens_sum
                    


                branch_mean_max = np.argmax(dens_mean)

                filament_x[str(num)] = copy.deepcopy(filament_branch_x[str(branch_mean_max)])
                filament_y[str(num)] = copy.deepcopy(filament_branch_y[str(branch_mean_max)])
                filament_z[str(num)] = copy.deepcopy(filament_branch_z[str(branch_mean_max)])

                previous_x = filament_x[str(num)][-1]
                previous_y = filament_y[str(num)][-1]
                previous_z = filament_z[str(num)][-1]
                
                break

        filament_zip = list(zip(filament_x[str(num)],filament_y[str(num)],filament_z[str(num)]))
        filament[str(num)] = filament_zip
        num = num + 1

    if os.path.isdir(save_path + cluster_num[:-4] + '/'):
        shutil.rmtree(save_path + cluster_num[:-4] + '/')
        
    os.makedirs(save_path + cluster_num[:-4] + '/')                    
    

    tmp_filament = []
    for i in list(filament.keys()):
        if len(filament[str(i)]) >= int(2/mpc_grid): # 5Mpc
            tmp_filament.append(filament[str(i)])
    
    
    if tmp_filament != []:
        save_candidate_filament = sorted(tmp_filament,key=len,reverse=True)
                
        del_list = []
        for i in range(np.array(save_candidate_filament).shape[0]-1):
            
            ref_filament = save_candidate_filament[i]
            
            for j in range(i+1,np.array(save_candidate_filament).shape[0]):
                
                com_filament = save_candidate_filament[j]

                count = 0
                for n,(com_x,com_y,com_z) in enumerate(np.array(com_filament)):

                    for nn in range(np.array(com_filament).shape[0]):

                        ref_x = np.array(ref_filament)[nn,0]
                        ref_y = np.array(ref_filament)[nn,1]
                        ref_z = np.array(ref_filament)[nn,2]

                        if com_x in np.linspace(ref_x-1,ref_x+1,3) and com_y in np.linspace(ref_y-1,ref_y+1,3) and com_z in np.linspace(ref_z-1,ref_z+1,3):
                            count = count + 1
                            
                
                
                if (count / np.array(com_filament).shape[0]) >= 0.5:
                    del_list.append(j)
                 

        for nn,n in enumerate(np.setdiff1d(np.linspace(0,np.array(tmp_filament).shape[0]-1,np.array(tmp_filament).shape[0]),np.unique(del_list))):
            np.savetxt(save_path + cluster_num[:-4] + '/' + str(int(nn)+1),tmp_filament[int(n)],fmt='%3i')






#%%