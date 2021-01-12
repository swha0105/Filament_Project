#%%
import numpy as np
import sys 
import os
import matplotlib.pyplot as plt
import copy 
import cv2
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
box_name = '02'
subbox_name = '2'
peak_name = '0'

path = '/storage/filament/result/cluster_3d/40Mpc/density_temp/box' + box_name + '/subbox0' +  subbox_name 
save_path = '/storage/filament/result/cluster_3d/40Mpc/density_temp/box' + box_name + '/subbox0' + subbox_name + '/filament/' + peak_name + '/'                             
data_type = ['a','f','g']
        


filament_conneted = np.fromfile(path + '/label/26/' + str(peak_name) ,dtype='int32')
filament_conneted = filament_conneted.reshape([101,101,101])

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

#%%            

filament = {}    
num = 0

filament_x = {}
filament_y = {}
filament_z = {}
# new filament!


for n in range(coords.shape[0]):
#for n in range(1):

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


            if not os.path.isdir(save_path  + '/'):
                os.makedirs(save_path + '/')
            os.chdir(save_path + '/')    

            for i in range(len(filament)):
                tmp_filament = filament[str(i)]
            
                np.savetxt(str(i),tmp_filament,fmt='%3i')

            
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
  
            for num_filament_candidate,(previous_x,previous_y,previous_z) in enumerate(candidate_coords_filament):
 
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

                        
            # 필라멘트 브런치들의 길이 테스트 한 뒤 밀도 큰걸로.
            dens_mean = np.zeros(candidate_coords_filament.shape[0])
            

            for num_filament_candidate in range(candidate_coords_filament.shape[0]):
                x_coords = filament_brunch_x[str(num_filament_candidate)]
                y_coords = filament_brunch_y[str(num_filament_candidate)]
                z_coords = filament_brunch_z[str(num_filament_candidate)]
                
                dens_sum = 0

                if len(x_coords) - len(filament_x[str(num)]) <=1:
                    for nn in range(len(x_coords)-1):
                        dens_sum = dens_sum + dens[x_coords[nn],y_coords[nn],z_coords[nn]]

                    dens_mean[num_filament_candidate] = (dens_sum + \
                        np.sum(dens[x_coords[-1]-1:x_coords[-1]+2,y_coords[-1]-1:y_coords[-1]+2,z_coords[-1]-1:z_coords[-1]+2]))/ \
                            (len(x_coords)-1 + 27)

                else:

                    for nn in range(len(x_coords)):
                        dens_sum = dens_sum + dens[x_coords[nn],y_coords[nn],z_coords[nn]]

                    dens_mean[num_filament_candidate] = dens_sum/len(x_coords)
                
            brunch_mean_max = np.argmax(dens_mean)


            filament_x[str(num)] = filament_brunch_x[str(brunch_mean_max)]
            filament_y[str(num)] = filament_brunch_y[str(brunch_mean_max)]
            filament_z[str(num)] = filament_brunch_z[str(brunch_mean_max)]

            previous_x = filament_x[str(num)][-1]
            previous_y = filament_y[str(num)][-1]
            previous_z = filament_z[str(num)][-1]

            #print(previous_x,previous_y,previous_z)
            # 브런치의 첫번째 좌표 반환.

                    


         
        


#%%%

# In[16]:


box_list = ['box01','box02']
subbox_list = ['subbox01','subbox02','subbox03','subbox04','subbox05','subbox06','subbox07','subbox08']

for box_name in box_list:
    for subbox_name in subbox_list:
        path = '/storage/filament/result/cluster_3d/40Mpc/density_temp/' + box_name + '/' +  subbox_name + '/label/'
        for peak_name in np.sort(os.listdir(path)):
            test = np.fromfile(path + str(peak_name) ,dtype='int32')
            test_re = test.reshape([101,101,101])
            box_size = 8 
            center_point = 50

            coords_x = []
            coords_y = []
            coords_z = []

            # x = 42
            plane = test_re[center_point - box_size, center_point-box_size : center_point+box_size  ,center_point-box_size : center_point+box_size ]
            coords = np.array(np.where(plane==1))

            for n in range(coords.shape[1]):
                coords_1 = center_point + coords[0,n] - box_size
                coords_2 = center_point + coords[1,n] - box_size 

                coords_x.append(center_point-box_size)
                coords_y.append(coords_1)
                coords_z.append(coords_2)

            # x = 58 
            plane = test_re[center_point + box_size, center_point-box_size : center_point+box_size  ,center_point-box_size : center_point+box_size ]
            coords = np.array(np.where(plane==1))

            for n in range(coords.shape[1]):
                coords_1 = center_point + coords[0,n] - box_size
                coords_2 = center_point + coords[1,n] - box_size 

                coords_x.append(center_point+box_size)
                coords_y.append(coords_1)
                coords_z.append(coords_2)

            # y = 42
            plane = test_re[center_point-box_size : center_point+box_size,center_point - box_size  ,center_point-box_size : center_point+box_size ]
            coords = np.array(np.where(plane==1))

            for n in range(coords.shape[1]):
                coords_1 = center_point + coords[0,n] - box_size
                coords_2 = center_point + coords[1,n] - box_size 

                coords_x.append(coords_1)
                coords_y.append(center_point-box_size)
                coords_z.append(coords_2)    

            # y = 58
            plane = test_re[center_point-box_size : center_point+box_size,center_point + box_size  ,center_point-box_size : center_point+box_size ]
            coords = np.array(np.where(plane==1))

            for n in range(coords.shape[1]):
                coords_1 = center_point + coords[0,n] - box_size
                coords_2 = center_point + coords[1,n] - box_size 

                coords_x.append(coords_1)
                coords_y.append(center_point+box_size)
                coords_z.append(coords_2)       

            # z = 42
            plane = test_re[center_point-box_size : center_point+box_size,center_point-box_size : center_point+box_size,center_point - box_size  ]
            coords = np.array(np.where(plane==1))

            for n in range(coords.shape[1]):
                coords_1 = center_point + coords[0,n] - box_size
                coords_2 = center_point + coords[1,n] - box_size 

                coords_x.append(coords_1)
                coords_y.append(coords_2)
                coords_z.append(center_point-box_size)    

            # z = 58
            plane = test_re[center_point-box_size : center_point+box_size,center_point-box_size : center_point+box_size,center_point + box_size]
            coords = np.array(np.where(plane==1))

            for n in range(coords.shape[1]):
                coords_1 = center_point + coords[0,n] - box_size
                coords_2 = center_point + coords[1,n] - box_size 

                coords_x.append(coords_1)
                coords_y.append(coords_2)
                coords_z.append(center_point+box_size)   
                
            coords = np.array(list(zip(coords_x,coords_y,coords_z)))
            
            
            
            ###
            
            filament = ()    

            for n in range(coords.shape[0]):
                filament_x = []
                filament_y = []
                filament_z = []

                ref_x = coords[n,0]
                ref_y = coords[n,1]
                ref_z = coords[n,2]

                previous_x = ref_x
                previous_y = ref_y
                previous_z = ref_z

                filament_x.append(previous_x)
                filament_y.append(previous_y)
                filament_z.append(previous_z)
                ### 필라멘트 1개의 integration
                for i in range(50):

                    test_box = test_re[previous_x-1:previous_x+2,previous_y-1:previous_y+2,previous_z-1:previous_z+2]

                    candidate_set = np.array(np.where(test_box==1)).T
                    
                    candidate_set = candidate_set - 1
                    if candidate_set.shape[0] <= 2:
                        break
                        candidate_set 
                    # #print(candidate_set.shape)
                    # if candidate_set.shape[0] == 0:
                    #     break
                    # elif candidate_set.shape[0] == 3:
                    #     candidate_x_list = []
                    #     candidate_y_list = []
                    #     candidate_z_list = []

                    #     distance_list = []
                    #     ### 박스
                    #     for j in range(candidate_set.shape[0]):

                    #         candidate_x = previous_x + (candidate_set[j][0]-1)
                    #         candidate_y = previous_y + (candidate_set[j][1]-1)
                    #         candidate_z = previous_z + (candidate_set[j][2]-1)

                    #         candidate_x_list.append(candidate_x)
                    #         candidate_y_list.append(candidate_y)
                    #         candidate_z_list.append(candidate_z)

                    #         distance_list.append(np.sqrt( (center_point - candidate_x)**2  + (center_point - candidate_y)**2 + (center_point - candidate_z)**2))

                    #     forward_point = np.argmax(distance_list)   

                    #     tmp_x = candidate_x_list[forward_point]
                    #     tmp_y = candidate_y_list[forward_point]
                    #     tmp_z = candidate_z_list[forward_point]

                    #     if tmp_x == previous_x and tmp_y == previous_y and tmp_z == previous_z:
                    #         break
                    #     else:
                    #         previous_x = tmp_x
                    #         previous_y = tmp_y
                    #         previous_z = tmp_z

                    #         filament_x.append(previous_x)
                    #         filament_y.append(previous_y)
                    #         filament_z.append(previous_z)
                    #     #else:
                    # elif candidate_set.shape[3] > 3:


                        

                filament_zip = list(zip(filament_x,filament_y,filament_z))
                filament = filament + (n,(filament_zip))
                
                save_path = '/storage/filament/result/cluster_3d/40Mpc/density_temp/' + box_name + '/' + subbox_name + '/filament/' + peak_name + '/'                             


                for i in range(len(filament)):
                    tmp = filament[i]

                    if type(tmp) == int:
                        if not os.path.isdir(save_path  + '/'):
                            os.makedirs(save_path + '/')
                        os.chdir(save_path + '/')    
                        tmp_filament = filament[i+1]
                        np.savetxt(str(tmp),tmp_filament,fmt='%3i')
                    else:
                        continue

        
    


# In[15]:


save_path



# In[17]:


box_list = ['box01_add','box02_add']

for box_name in box_list:
    box_path = '/storage/filament/result/cluster_3d/40Mpc/density_temp/' + box_name + '/'
    for subbox_name in os.listdir(box_path):
        path = '/storage/filament/result/cluster_3d/40Mpc/density_temp/' + box_name + '/' +  subbox_name + '/label/'
        for peak_name in np.sort(os.listdir(path)):
            test = np.fromfile(path + '0' ,dtype='int32')
            test_re = test.reshape([101,101,101])
            box_size = 8 
            center_point = 50

            coords_x = []
            coords_y = []
            coords_z = []

            # x = 42
            plane = test_re[center_point - box_size, center_point-box_size : center_point+box_size  ,center_point-box_size : center_point+box_size ]
            coords = np.array(np.where(plane==1))

            for n in range(coords.shape[1]):
                coords_1 = center_point + coords[0,n] - box_size
                coords_2 = center_point + coords[1,n] - box_size 

                coords_x.append(center_point-box_size)
                coords_y.append(coords_1)
                coords_z.append(coords_2)

            # x = 58 
            plane = test_re[center_point + box_size, center_point-box_size : center_point+box_size  ,center_point-box_size : center_point+box_size ]
            coords = np.array(np.where(plane==1))

            for n in range(coords.shape[1]):
                coords_1 = center_point + coords[0,n] - box_size
                coords_2 = center_point + coords[1,n] - box_size 

                coords_x.append(center_point+box_size)
                coords_y.append(coords_1)
                coords_z.append(coords_2)

            # y = 42
            plane = test_re[center_point-box_size : center_point+box_size,center_point - box_size  ,center_point-box_size : center_point+box_size ]
            coords = np.array(np.where(plane==1))

            for n in range(coords.shape[1]):
                coords_1 = center_point + coords[0,n] - box_size
                coords_2 = center_point + coords[1,n] - box_size 

                coords_x.append(coords_1)
                coords_y.append(center_point-box_size)
                coords_z.append(coords_2)    

            # y = 58
            plane = test_re[center_point-box_size : center_point+box_size,center_point + box_size  ,center_point-box_size : center_point+box_size ]
            coords = np.array(np.where(plane==1))

            for n in range(coords.shape[1]):
                coords_1 = center_point + coords[0,n] - box_size
                coords_2 = center_point + coords[1,n] - box_size 

                coords_x.append(coords_1)
                coords_y.append(center_point+box_size)
                coords_z.append(coords_2)       

            # z = 42
            plane = test_re[center_point-box_size : center_point+box_size,center_point-box_size : center_point+box_size,center_point - box_size  ]
            coords = np.array(np.where(plane==1))

            for n in range(coords.shape[1]):
                coords_1 = center_point + coords[0,n] - box_size
                coords_2 = center_point + coords[1,n] - box_size 

                coords_x.append(coords_1)
                coords_y.append(coords_2)
                coords_z.append(center_point-box_size)    

            # z = 58
            plane = test_re[center_point-box_size : center_point+box_size,center_point-box_size : center_point+box_size,center_point + box_size]
            coords = np.array(np.where(plane==1))

            for n in range(coords.shape[1]):
                coords_1 = center_point + coords[0,n] - box_size
                coords_2 = center_point + coords[1,n] - box_size 

                coords_x.append(coords_1)
                coords_y.append(coords_2)
                coords_z.append(center_point+box_size)   
                
            coords = np.array(list(zip(coords_x,coords_y,coords_z)))
            
            
            
            ###
            
            filament = ()    

            for n in range(coords.shape[0]):
                filament_x = []
                filament_y = []
                filament_z = []

                ref_x = coords[n,0]
                ref_y = coords[n,1]
                ref_z = coords[n,2]

                previous_x = ref_x
                previous_y = ref_y
                previous_z = ref_z

                filament_x.append(previous_x)
                filament_y.append(previous_y)
                filament_z.append(previous_z)
                ### 필라멘트 1개의 integration
                for i in range(50):

                    test_box = test_re[previous_x-1:previous_x+2,previous_y-1:previous_y+2,previous_z-1:previous_z+2]

                    candidate_set = np.array(np.where(test_box==1)).T
                    #print(candidate_set.shape)
                    if candidate_set.shape[0] ==0:
                        break
                    else:
                        candidate_x_list = []
                        candidate_y_list = []
                        candidate_z_list = []

                        distance_list = []
                        ### 박스
                        for j in range(candidate_set.shape[0]):

                            candidate_x = previous_x + (candidate_set[j][0]-1)
                            candidate_y = previous_y + (candidate_set[j][1]-1)
                            candidate_z = previous_z + (candidate_set[j][2]-1)

                            candidate_x_list.append(candidate_x)
                            candidate_y_list.append(candidate_y)
                            candidate_z_list.append(candidate_z)

                            distance_list.append(np.sqrt( (center_point - candidate_x)**2  + (center_point - candidate_y)**2 + (center_point - candidate_z)**2))

                        forward_point = np.argmax(distance_list)   

                        tmp_x = candidate_x_list[forward_point]
                        tmp_y = candidate_y_list[forward_point]
                        tmp_z = candidate_z_list[forward_point]

                        if tmp_x == previous_x and tmp_y == previous_y and tmp_z == previous_z:
                            break
                        else:
                            previous_x = tmp_x
                            previous_y = tmp_y
                            previous_z = tmp_z

                            filament_x.append(previous_x)
                            filament_y.append(previous_y)
                            filament_z.append(previous_z)
                        #else:


                filament_zip = list(zip(filament_x,filament_y,filament_z))
                filament = filament + (n,(filament_zip))
                
                save_path = '/storage/filament/result/cluster_3d/40Mpc/density_temp/' + box_name + '/' + subbox_name + '/filament/'                              


                for i in range(len(filament)):
                    tmp = filament[i]

                    if type(tmp) == int:
                        if not os.path.isdir(save_path  + '/'):
                            os.makedirs(save_path + '/')
                        os.chdir(save_path + '/')    
                        tmp_filament = filament[i+1]
                        np.savetxt(str(tmp),tmp_filament,fmt='%3i')
                    else:
                        continue

        
        

