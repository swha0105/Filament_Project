#%%
import copy
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import gc
from numba import jit,njit, float32,int32

#%%

@jit((int32(float32,int32,int32)))
def local_maximum(xray,vr,const):
    candidate_coordx = []
    candidate_coordy = []
    candidate_coordz = []
    
    
    for iz in range(vr,xray.shape[0]-vr):
        for iy in range(vr,xray.shape[0]-vr):
            for ix in range(vr,xray.shape[0]-vr):                
                if xray[ix,iy,iz]*const >= -5.5:
                    

                    ref_value = xray[ix,iy,iz]*const
                    tmp_x = ix
                    tmp_y = iy
                    tmp_z = iz
                    max_value = ref_value
                    for rx in range(-vr,vr+1):
                        for ry in range(-vr,vr+1):
                            for rz in range(-vr,vr+1):
                                if ref_value < xray[ix+rx, iy+ry, iz+rz]*const:
                                    tmp_x = 0
                                    tmp_y = 0
                                    tmp_z = 0


                                else:
                                    pass

                    if tmp_x != 0 and tmp_y != 0 and tmp_z != 0:
                            
                        candidate_coordx.append(tmp_x)
                        candidate_coordy.append(tmp_y)
                        candidate_coordz.append(tmp_z)
    

    coords = np.array(list(zip(candidate_coordx,candidate_coordy,candidate_coordz)),np.int32)
    
    return coords





# %%
res = 2048
box_list = ['1']
box_length = '300Mpc'
q = ['xray']
cluster_length = 45
volume_const  = 0.01*( np.float32(int(box_length[:3])/2048)*(3.086/0.7))**3

if box_length == '100Mpc':
    cluster_grid = 900   #45Mpc, Smootinhg: 12, 0.586 Grid/Mpc   
    mpc_grid = 0.05
if box_length == '200Mpc':  
    cluster_grid = 450   #45Mpc, Smoothing: 6, 0.586 Grid/Mpc  
    mpc_grid = 0.1
if box_length == '300Mpc':
    cluster_grid = 300   #45Mpc, Smoothing: 4, 0.586 Grid/Mpc 
    mpc_grid = 0.15  

#(smootinhg scale * 75  )
# for box_num in box_list:
box_num = '1'
path = '/storage/filament/works_v4/data/' + box_length + '_' + box_num + '/'

# for q_name in q:
q_name = 'xray'
xray = np.fromfile(path + 'raw/' + 'L' + box_length[:3] + q_name,dtype=np.float32).reshape([res,res,res])
#%%
radius_for_local_maximum = int(2/mpc_grid)
coords = local_maximum(xray,radius_for_local_maximum,volume_const)




#%%


# %%
res = 2048
box_list = ['1','2']
box_length = '300Mpc'
q = ['den']
cluster_length = 45

if box_length == '100Mpc':
    cluster_grid = 900   #45Mpc, Smootinhg: 12, 0.586 Grid/Mpc   
if box_length == '200Mpc':  
    cluster_grid = 450   #45Mpc, Smoothing: 6, 0.586 Grid/Mpc  
if box_length == '300Mpc':
    cluster_grid = 300   #45Mpc, Smoothing: 4, 0.586 Grid/Mpc   

#(smootinhg scale * 75  )
for box_num in box_list:
    path = '/storage/filament/works_v4/data/' + box_length + '_' + box_num + '/'

    coords = np.loadtxt(path + 'raw/' + 'Cinfo.dat')[:,1:]
    coords = coords[coords[:,3].argsort()]
    temp_coords = coords[:,3]
    coords = coords[:,:3]

    for q_name in q:
        dens = np.fromfile(path + 'raw/' + 'L' + box_length[:3] + q_name,dtype=np.float32).reshape([res,res,res])

        sort_x = []
        sort_y = []
        sort_z = []
        for n,(iz,iy,ix) in enumerate(coords):

            ix = int(ix)
            iy = int(iy)
            iz = int(iz)

            if temp_coords[n] < 2.0 or temp_coords[n] > 3.0:
                continue
            else:
                    
                ref_x = ix
                ref_y = iy
                ref_z = iz 

                if ref_x < int(cluster_grid/2):
                    ref_tmp_x = res + ref_x
                else:
                    ref_tmp_x = ref_x
                if ref_y < int(cluster_grid/2):
                    ref_tmp_y = res + ref_y
                else:
                    ref_tmp_y = ref_y
                if ref_z < int(cluster_grid/2):
                    ref_tmp_z = res + ref_z
                else:
                    ref_tmp_z = ref_z


                for _,(iiz,iiy,iix) in enumerate(coords[n+1:,:]):
                    
                    
                    iix = int(iix)
                    iiy = int(iiy)
                    iiz = int(iiz)
                    
                    if iix < int(cluster_grid/2):
                        tmp_x = res + iix
                    else:
                        tmp_x = iix
                    if iiy < int(cluster_grid/2):
                        tmp_y = res + iiy
                    else:
                        tmp_y = iiy
                    if iiz < int(cluster_grid/2):
                        tmp_z = res + iiz
                    else:
                        tmp_z = iiz

                    if np.sqrt((ref_tmp_x-tmp_x)**2 + (ref_tmp_y-tmp_y)**2 + (ref_tmp_z-tmp_z)**2) < int(cluster_grid/2) or \
                    np.sqrt((ix-iix)**2 + (iy-iiy)**2 + (iz-iiz)**2) < int(cluster_grid/2):

                        ref_x = 0 
                        ref_y = 0 
                        ref_z = 0 
                
                if ref_x != 0 and ref_y != 0 and ref_z != 0 :
                    
                    sort_x.append(ref_x)
                    sort_y.append(ref_y)
                    sort_z.append(ref_z)

        coords_set = np.array(list(set(zip(sort_x,sort_y,sort_z))))
        print(coords_set.shape)       

        if not os.path.isdir(path + 'DL/raw_data/' + q_name +'/'):
            os.makedirs(path + 'DL/raw_data/' + q_name +'/')
        cluster_save_path = path + '/DL/raw_data/' + q_name + '/'

        np.savetxt(path + 'DL/raw_data/cluster_info',np.hstack([coords_set,np.expand_dims( np.linspace(1,len(coords_set),len(coords_set)),axis=1)]),fmt='%3i' )



        if not os.path.isdir(cluster_save_path):
            os.makedirs(cluster_save_path)

        for cluster_num,(ix,iy,iz) in enumerate(coords_set):
            cluster_dens = []
            
            print(cluster_num)
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

                        cluster_dens.append(np.log10(dens[iix,iiy,iiz]))

            cluster_dens = np.array(cluster_dens).reshape([cluster_grid,cluster_grid,cluster_grid])


            np.save(cluster_save_path + str(cluster_num+1),cluster_dens)

        del dens
        gc.collect()        