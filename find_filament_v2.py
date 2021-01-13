#%%
import copy
import os
import sys
import shutil
import cv2 
import matplotlib.pyplot as plt
import numpy as np
import copy
import networkx as nx
    

skeleton_path = '/storage/filament/works_v7/300Mpc_1/label/skeleton/'

for cluster_num in os.listdir(skeleton_path):
    #cluster_num = '3.npy'   
    if cluster_num == '36.npy':
        continue
    

    skeleton = np.load(skeleton_path + cluster_num)

    filament_end_point = np.array(np.where(skeleton==2)).T


    coords_x = []
    coords_y = []
    coords_z = []

    center_point = int(skeleton.shape[0]/2)

    index = np.argwhere(skeleton!=0)
    graph = {}
    for n,(ix,iy,iz) in enumerate(index):
        
        candidate_set = np.argwhere(skeleton[ix-1:ix+2,iy-1:iy+2,iz-1:iz+2] != 0) - 1 

        graph[str(n)] = []
        for iix,iiy,iiz in candidate_set:
            if iix == 0 and iiy == 0 and iiz == 0:
                continue
            else:
                for num,(tmp_x,tmp_y,tmp_z) in enumerate(index):
                    if tmp_x == ix+iix and tmp_y == iy+iiy and tmp_z == iz+iiz:
                        graph[str(n)].append(str(num))

    g = nx.DiGraph(graph)
    un_g = g.to_undirected()


    print(cluster_num)


    start_index = str(np.argmin(np.sum(np.abs(index-150),axis=1)))

    for filament_num,tmp in enumerate(filament_end_point):

        end_x = tmp[0]
        end_y = tmp[1]
        end_z = tmp[2]

        end_index = str(np.where((index[:,0]==end_x) & (index[:,1] == end_y) & (index[:,2] == end_z))[0][0])
        end_coords = index[int(end_index)]

        coords_index = list(nx.all_simple_paths(un_g,source=start_index,target=end_index,cutoff=300))
        
        
        if coords_index == []:
            continue

        
        for num_filament_candidate in range(np.array(coords_index).shape[0]):
            filament_list = []
            for filament_index in coords_index[num_filament_candidate]:
                filament_list.append(list(index[int(filament_index)]))

            
            if not os.path.isdir('/storage/filament/works_v7/300Mpc_1/filament/candidate/' + cluster_num[:-4] +'/' + str(filament_num+1) + '/'):
                os.makedirs('/storage/filament/works_v7/300Mpc_1/filament/candidate/' + cluster_num[:-4] +'/' + str(filament_num+1) + '/' )
            np.savetxt('/storage/filament/works_v7/300Mpc_1/filament/candidate/' + cluster_num[:-4] + '/'  + str(filament_num+1) + '/' + str(num_filament_candidate+1),filament_list,fmt="%i")
            
            
    

#%%

path = '/storage/filament/works_v7/300Mpc_1/filament/38/2/'
filament_list = []
for filament_num in np.sort(np.array(os.listdir(path))):
    tmp = list(np.loadtxt(path + filament_num))
    filament_list.append(tmp)


#%%

max_length = 0
for i in range(np.array(filament_list).shape[0]):
    max_length = np.max([max_length,np.array(filament_list[i]).shape[0]])

#%%
new_k = []
for elem in filament_list:
    elem_list = np.sort(elem).tolist()
    if elem_list not in new_k:
        new_k.append(elem_list)

print(np.array(new_k).shape)

#%%

k = [[1, 2], [4], [5, 6, 2], [1, 2], [3], [4]]
new_k = []
for elem in k:
    if elem not in new_k:
        new_k.append(elem)
k = new_k
print(k)
#%%



#L_dict = dict((x[0], x[1:]) for x in filament_list)

