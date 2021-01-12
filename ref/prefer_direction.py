#%%
import copy
import os
import sys
import matplotlib.pyplot as plt
import numpy as np



def all_equal(iterator):
  try:
     iterator = iter(iterator)
     first = next(iterator)
     return all(np.array_equal(first, rest) for rest in iterator)
  except StopIteration:
     return True


        
#%%

ref_path = '/storage/filament/works_v7/300Mpc_1/filament/candidate/'
save_path = '/storage/filament/works_v7/300Mpc_1/filament/sorted/'
xray_path = '/storage/filament/works_v7/300Mpc_1/cluster_box/xray/'
#ref_norm = 10
#direction_norm = 10

filament_candidate = []
length_list = []

for cluster_num in np.sort(np.array(os.listdir(ref_path))):
    if  cluster_num == '36.npy':
        continue

    cluster_path = ref_path + cluster_num + '/'
    xray = np.load('/storage/filament/works_v7/300Mpc_1/cluster_box/xray/' + cluster_num + '.npy')
    for filament_num in np.sort(np.array(os.listdir(cluster_path))):

        print(cluster_num,filament_num)
        filament_path = cluster_path + filament_num + '/'

        filament_candidate = []
        length_list = []

        if np.array(os.listdir(filament_path)).shape[0] == 1:
            if not os.path.isdir(save_path + cluster_num + '/' ):
                os.makedirs(save_path + cluster_num + '/' )
            np.savetxt(save_path + cluster_num + '/' + filament_num, np.loadtxt(filament_path + '1',dtype=np.int16) )
            continue

        else:
            for candidate_num in np.sort(np.array(os.listdir(filament_path))):

                filament_candidate.append(np.loadtxt(filament_path + candidate_num))
                length_list.append(len(np.loadtxt(filament_path + candidate_num,dtype=np.int16)))

            direction_vector = np.zeros([ len(filament_candidate),int(np.min(length_list)-10),3 ])
            branch_vector = np.zeros([ len(filament_candidate),3])
            xray_sum = np.zeros([len(filament_candidate)])
            direction_vector_list = []
            tmp_vector = []

            index = 999

            for i in range(int(np.min(length_list))-10):

                for candidate_num in range(len(filament_candidate)):
                    direction_vector[candidate_num,i,:] = filament_candidate[candidate_num][i+1]  - filament_candidate[candidate_num][i]  
                    
                if all_equal(direction_vector[:,i,:]) == True: 
                    continue

                else:
                    # 현재까지  vector sum
                    #refer_vector = np.sum(direction_vector_list,axis=0)/np.sqrt(np.sum(np.abs(direction_vector_list),axis=0)**2)
                    tmp = filament_candidate[0][-1] - filament_candidate[0][i]
                    refer_vector = tmp/np.sqrt(np.sum(tmp**2))

                    tmp = 0
                    
                    for j in range(len(filament_candidate)):
                        for k in range(i,i+10):
                            
                            tmp = tmp + (filament_candidate[j][k+1]  - filament_candidate[j][k])
                            tmp_xray = np.array(filament_candidate[j][k],dtype=np.int16)
                            xray_sum[j] = xray_sum[j] + xray[tmp_xray[0],tmp_xray[1],tmp_xray[2]]
                            # print(xray_sum[j])
                            #print(np.sqrt(np.sum(tmp**2)),tmp,tmp**2,tmp*tmp )        
                        
                        branch_vector[j,:] = (tmp/np.sqrt(np.sum(tmp**2)))
                            
                    for ix in range(len(filament_candidate)):
                        tmp_vector.append(np.dot(refer_vector,branch_vector[ix]))


                    tmp_vector = tmp_vector*xray_sum
                    index = np.argmax(tmp_vector)
                    break
                    
        
        
            if not os.path.isdir(save_path + cluster_num + '/' ):
                os.makedirs(save_path + cluster_num + '/')
            np.savetxt(save_path + cluster_num + '/' + filament_num,np.array(filament_candidate)[index])

