#%%
import copy
import os
import sys
import shutil
import cv2 
import matplotlib.pyplot as plt
import numpy as np

#%%

def find_candidate_coords(previous_x,previous_y,previous_z,filament_x,filament_y,filament_z,num):
    test_box = filament_connected[previous_x-1:previous_x+2,previous_y-1:previous_y+2,previous_z-1:previous_z+2]
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
box_length = '200Mpc'
resolution = 1024
box_num = '1'
distance = 0 

if box_length == '200Mpc':
    if resolution == 1024:
        distance = 100
    elif resolution == 512:
        distance = 50
elif box_length == '300Mpc':
    if resolution == 1024:
        distance = 70
    elif resolution == 512:
        distance = 35
elif box_length == '100Mpc':
    if resolution == 1024:
        distance = 200
    elif resolution == 512:
        distance = 100


data_path = '/storage/filament/works_v2/data_filament/' + box_length + '_' + box_num + '_' + str(resolution) + '/clusters/'
result_path = '/storage/filament/works_v2/result_filament/'  + box_length + '_' + box_num + '_' + str(resolution) + '/'
 

res = distance*2 + 1 
box_size = int(distance/5)
#box_size = 20
center_point = distance-1

dens = np.zeros([res,res,res])
#filament_tmp = np.zeros([res-2,res-2,res-2])
filament_conneted = np.zeros([res-2,res-2,res-2])
filament_major = np.zeros([res,res,res])


#for cluster_num in np.array(sorted(os.listdir(data_path),key=int)):
#for cluster_num in range(1,2):    
    
#cluster_num = str(cluster_num)
cluster_num = str(1)
dens_path = data_path + cluster_num + '/dens.npy'


dens = np.load(dens_path).reshape([res,res,res])

#%%

data_type = ['a','f','g']


filament_tmp = np.fromfile('/storage/filament/works_v3/nexus/200Mpc_1_1024/1/post_processing',dtype='int32')
filament_connected = filament_tmp.reshape([res-2,res-2,res-2],order='F')
#%%


#dens = np.load(dens_path).reshape([res,res,res])

coords_x = []
coords_y = []
coords_z = []

#x = 42
plane = filament_connected[center_point - box_size, center_point-box_size : center_point+box_size+1  ,center_point-box_size : center_point+box_size+1 ]
coords = np.array(np.where(plane==1))

for n in range(coords.shape[1]):
    coords_1 = center_point + coords[0,n] - box_size
    coords_2 = center_point + coords[1,n] - box_size 

    coords_x.append(center_point-box_size)
    coords_y.append(coords_1)
    coords_z.append(coords_2)

# x = 58 
plane = filament_connected[center_point + box_size, center_point-box_size : center_point+box_size+1  ,center_point-box_size : center_point+box_size+1 ]
coords = np.array(np.where(plane==1))

for n in range(coords.shape[1]):
    coords_1 = center_point + coords[0,n] - box_size
    coords_2 = center_point + coords[1,n] - box_size 

    coords_x.append(center_point+box_size)
    coords_y.append(coords_1)
    coords_z.append(coords_2)

# y = 42
plane = filament_connected[center_point-box_size : center_point+box_size+1, center_point - box_size  ,center_point-box_size : center_point+box_size+1 ]
coords = np.array(np.where(plane==1))

for n in range(coords.shape[1]):
    coords_1 = center_point + coords[0,n] - box_size
    coords_2 = center_point + coords[1,n] - box_size 

    coords_x.append(coords_1)
    coords_y.append(center_point-box_size)
    coords_z.append(coords_2)    

# y = 58
plane = filament_connected[center_point-box_size : center_point+box_size+1, center_point + box_size  ,center_point-box_size : center_point+box_size+1 ]
coords = np.array(np.where(plane==1))

for n in range(coords.shape[1]):
    coords_1 = center_point + coords[0,n] - box_size
    coords_2 = center_point + coords[1,n] - box_size 

    coords_x.append(coords_1)
    coords_y.append(center_point+box_size)
    coords_z.append(coords_2)       

# z = 42
plane = filament_connected[center_point-box_size : center_point+box_size+1, center_point-box_size : center_point+box_size+1 , center_point - box_size  ]
coords = np.array(np.where(plane==1))

for n in range(coords.shape[1]):
    coords_1 = center_point + coords[0,n] - box_size
    coords_2 = center_point + coords[1,n] - box_size 

    coords_x.append(coords_1)
    coords_y.append(coords_2)
    coords_z.append(center_point-box_size)    

# z = 58
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

#%%

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

#%%
save_path = '/storage/filament/works_v3/nexus/200Mpc_1_1024/1/filament/'
if os.path.isdir(save_path):
    shutil.rmtree(save_path)
    
os.makedirs(save_path)                    


tmp_filament = []
for i in list(filament.keys()):
    tmp_filament.append(filament[str(i)])

save_filament = sorted(tmp_filament,key=len,reverse=True)

for i in list(filament.keys()):
    np.savetxt(save_path + str(int(i)+1),save_filament[int(i)],fmt='%3i')




# %%
