#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.semi_supervised import LabelSpreading
from numba import jit,njit, float32,int32
import os 
import copy

#@jit(float32(float32,float32))

def void_detect(label,temperature_raw):
    
    return_label = copy.deepcopy(label)

    for ix in range(return_label.shape[0]):
        for iy in range(return_label.shape[0]):
            for iz in range(return_label.shape[0]):
                if (temperature_raw[ix,iy,iz]) < 4:
                    return_label[ix,iy,iz] = 0

    return return_label


def wall_detect(label,temperature_raw):
    
    return_label = copy.deepcopy(label)

    for ix in range(return_label.shape[0]):
        for iy in range(return_label.shape[0]):
            for iz in range(return_label.shape[0]):
                if temperature_raw[ix,iy,iz] < 5 and temperature_raw[ix,iy,iz] >= 4:
                    return_label[ix,iy,iz] = 1

    return return_label    


@njit
def local_maximum(xray,vr):
    candidate_coordx = []
    candidate_coordy = []
    candidate_coordz = []
    

    for iz in range(vr,xray.shape[0]-vr):
        for iy in range(vr,xray.shape[0]-vr):
            for ix in range(vr,xray.shape[0]-vr):                
                if xray[ix,iy,iz] >= -6.0:
                    

                    ref_value = xray[ix,iy,iz]
                    tmp_x = ix
                    tmp_y = iy
                    tmp_z = iz
                    max_value = ref_value
                    for rx in range(-vr,vr+1):
                        for ry in range(-vr,vr+1):
                            for rz in range(-vr,vr+1):
                                if ref_value < xray[ix+rx, iy+ry, iz+rz]:
                                    tmp_x = 0
                                    tmp_y = 0
                                    tmp_z = 0


                                else:
                                    pass

                    if tmp_x != 0 and tmp_y != 0 and tmp_z != 0:
                            
                        candidate_coordx.append(tmp_x)
                        candidate_coordy.append(tmp_y)
                        candidate_coordz.append(tmp_z)

    coords = np.array(list(zip(candidate_coordx,candidate_coordy,candidate_coordz)))
    
    return coords


@jit(float32(float32,int32))
def smoothing_real_space(quantity,smoothing_scale):
    array_size = len(np.arange(0,cluster_grid,4))
    smoothing_array = np.zeros([array_size,array_size,array_size],dtype=np.float32)
    
    for ix,i in enumerate(np.arange(0,cluster_grid,4)):
        for iy,j in enumerate(np.arange(0,cluster_grid,4)):
            for iz,k in enumerate(np.arange(0,cluster_grid,4)):
                i = int(i)
                j = int(j)
                k = int(k)
           
                smoothing_array[ix,iy,iz] = np.mean(quantity[i:i+smoothing_scale,j:j+smoothing_scale,k:k+smoothing_scale])
    
    return smoothing_array

@jit(float32(float32,float32,int32))
def get_xray(dens,temp,box_length):
    box_len = int(box_length[:3])
    xray_re_normal = np.zeros([dens.shape[0],dens.shape[0],dens.shape[0]],dtype=np.float32)
    volume_const  = 0.01*( np.float32(box_len/2048)*(3.086/0.7))**3

    urho = 1.879*10**(-29) * (1+51.59113)**(3) * (0.7)**(2) * 0.28 
    fmton = ( urho /(1.67 * 10**(-24)) ) / (1+51.9113)**3
    rhoi = 0.76 + 0.24
    rhoe = 0.76 + 0.5*0.24
    const = 1.5*1.42*10**(3)*rhoe*rhoi*fmton**2

    dens_unnormal = (10**dens) * (0.044)/(0.28)

    xray_re_normal = np.log10(const*dens_unnormal**2 * np.sqrt(10**temp) * volume_const)
    return xray_re_normal


#%%
box_length = '300Mpc'
cluster_length = 45
box_list = ['1']

if box_length == '200Mpc':  
    cluster_grid = 450   #45Mpc, Smoothing: 6, 0.586 Grid/Mpc  
    smoothing_scale = 6
if box_length == '300Mpc':
    cluster_grid = 300   #45Mpc, Smoothing: 4, 0.586 Grid/Mpc   
    smoothing_scale = 4
if box_length == '100Mpc':
    cluster_grid = 900   #45Mpc, Smootinhg: 12, 0.586 Grid/Mpc   
    smoothing_scale = 12
for box_num in box_list:

    ref_path = '/storage/filament/works_v6/' + box_length + '_' + str(box_num) + '/'
    virial_radius_info = np.loadtxt(ref_path + 'cluster_box/virial_radius')
    #for cluster_num in np.sort(np.setdiff1d(np.array(os.listdir(ref_path + 'cluster_box/xray/')),np.array(os.listdir(ref_path + 'label/spreading/')))):
    for cluster_num in np.sort(np.array(os.listdir(ref_path + 'cluster_box/xray/'))):
        print(cluster_num)

        filament_signature = np.load(ref_path + 'signature/filament/' + cluster_num)
        cluster_signature = np.load(ref_path + 'signature/cluster/' + cluster_num)
        wall_signature = np.load(ref_path + 'signature/wall/' + cluster_num)
        
        xray = np.load(ref_path +  'cluster_box/xray/' + cluster_num)
        xray = xray[:filament_signature.shape[0],:filament_signature.shape[0],:filament_signature.shape[0]]
        temp = np.load(ref_path +  'cluster_box/temp/' + cluster_num)

        log_xray_range = np.linspace(np.min(xray),np.max(xray),100)

        label_raw = np.full([filament_signature.shape[0],filament_signature.shape[0],filament_signature.shape[0]],-1)
        label = void_detect(label_raw,temp)

        filament_signature = np.load(ref_path + 'signature/filament/' + cluster_num)
        cluster_signature = np.load(ref_path + 'signature/cluster/' + cluster_num)
        wall_signature = np.load(ref_path + 'signature/wall/' + cluster_num)
        
        xray = np.load(ref_path +  'cluster_box/xray/' + cluster_num)
        xray = xray[:filament_signature.shape[0],:filament_signature.shape[0],:filament_signature.shape[0]]
        temp = np.load(ref_path +  'cluster_box/temp/' + cluster_num)

        log_xray_range = np.linspace(np.min(xray),np.max(xray),100)

        label_raw = np.full([filament_signature.shape[0],filament_signature.shape[0],filament_signature.shape[0]],-1)
        label = void_detect(label_raw,temp)

        halo_info = np.loadtxt('/storage/filament/works_v6/300Mpc_1/data/group_info/' + cluster_num[:-4])

        label = wall_detect(label,temp)
        

        if len(halo_info.shape) <= 1:
            continue
        else:

            c_ix = int(halo_info[0,1])
            c_iy = int(halo_info[0,2])
            c_iz = int(halo_info[0,3])
            c_vr = int(np.around(halo_info[0,4]))

            for ix in range(int(label.shape[0]/2)-int(c_vr),int(label.shape[0]/2)+int(c_vr)+1):
                for iy in range(int(label.shape[0]/2)-int(c_vr),int(label.shape[0]/2)+int(c_vr)+1):
                    for iz in range(int(label.shape[0]/2)-int(c_vr),int(label.shape[0]/2)+int(c_vr)+1):
                        if np.sqrt( (ix-int(label.shape[0]/2))**2 + (iy-int(label.shape[0]/2))**2 + (iz-int(label.shape[0]/2))**2) <= int(c_vr):
                            label[ix,iy,iz] = 2

        
                for _,tmp in enumerate(halo_info[1:,:]):
                    h_ix = int(tmp[1])
                    h_iy = int(tmp[2])
                    h_iz = int(tmp[3])
                    h_vr = int(np.around(tmp[4]))


                    x_min = c_ix-h_ix+int(label.shape[0]/2) - h_vr
                    x_max = c_ix-h_ix+int(label.shape[0]/2) + h_vr + 1

                    y_min = c_iy-h_iy+int(label.shape[0]/2) - h_vr
                    y_max = c_iy-h_iy+int(label.shape[0]/2) + h_vr + 1

                    z_min = c_iz-h_iz+int(label.shape[0]/2) - h_vr
                    z_max = c_iz-h_iz+int(label.shape[0]/2) + h_vr + 1

                    for ix in range(x_min, x_max):
                        for iy in range(y_min,y_max):
                            for iz in range(z_min,z_max):
                                if np.sqrt( (ix- (x_min+h_vr))**2 + (iy- (y_min+h_vr))**2 + (iz-(z_min+h_vr))**2 ) <= int(h_vr) and ix < 271 and iy < 271 and iz < 271:
                                    if label[ix,iy,iz] == -1:
                                        label[ix,iy,iz] = 2
                                    else:
                                        continue
                                    
                                    
                                #   print("h")
                                
        
            index = np.argwhere(label!=0)


            # fraction_list = []

            # for threshold in log_xray_range:
                
            #     volume_count = (xray[xray>threshold]).shape[0]
            #     fraction_list.append(volume_count / (xray.shape[0]**3))

            #     if volume_count / (xray.shape[0]**3) < 0.24:
            #         print(cluster_num,threshold)
            #         break
            
            # threshold_1 = threshold

            # for threshold in log_xray_range:
                
            #     volume_count = (xray[xray>threshold]).shape[0]

            #     fraction_list.append(volume_count / (xray.shape[0]**3))
            #     if volume_count / (xray.shape[0]**3) < 0.1:
            #         print(cluster_num,threshold)
            #         break
            
            # threshold_2 = threshold

            label_re = copy.deepcopy(label)


            # for ix in range(filament_signature.shape[0]):
            #     for iy in range(filament_signature.shape[0]):
            #         for iz in range(filament_signature.shape[0]):
            #             if xray[ix,iy,iz] < threshold_1 and label[ix,iy,iz] == -1:
            #                 label_re[ix,iy,iz] = 1
                        
            # for ix in range(filament_signature.shape[0]):
            #     for iy in range(filament_signature.shape[0]):
            #         for iz in range(filament_signature.shape[0]):
            #             if xray[ix,iy,iz] > threshold_1 and xray[ix,iy,iz] < threshold_2 and label[ix,iy,iz] == -1:
            #                 label_re[ix,iy,iz] = 1
            #             elif xray[ix,iy,iz] > threshold_2 and label[ix,iy,iz] == -1:
            #                 label_re[ix,iy,iz] = -1


        label_refined = np.delete(label_re.flatten(),np.argwhere(label_re.flatten()==0))
        xray_refined = np.delete(xray.flatten(),np.argwhere(label_re.flatten()==0))
        cluster_refined = np.delete(cluster_signature.flatten(),np.argwhere(label_re.flatten()==0))
        filament_refined = np.delete(filament_signature.flatten(),np.argwhere(label_re.flatten()==0))
        wall_refined = np.delete(wall_signature.flatten(),np.argwhere(label_re.flatten()==0))


        data_flat = np.zeros([3,len(cluster_refined.flatten())])
        
        cluster_norm = (cluster_refined.flatten() - np.min(cluster_refined.flatten()) ) / ( np.max(cluster_refined.flatten()) - np.min(cluster_refined.flatten()))
        filament_norm = (filament_refined.flatten() - np.min(filament_refined.flatten()) ) / ( np.max(filament_refined.flatten()) - np.min(filament_refined.flatten()))
        wall_norm = (wall_refined.flatten() - np.min(wall_refined.flatten()) ) / ( np.max(wall_refined.flatten()) - np.min(wall_refined.flatten()))
        xray_norm = (xray_refined- np.min(xray_refined )) / ( np.max(xray_refined) - np.min(xray_refined))

        #data_flat[0,:] = cluster_norm
        data_flat[0,:] = filament_norm
        data_flat[1,:] = wall_norm
        data_flat[2,:] = xray_norm
        
        label_spread = LabelSpreading(kernel='knn', alpha=0.2,n_jobs=-1,n_neighbors=7)
                # n_neighbors 넣어야됨 

        label_spread.fit(data_flat.T, label_refined.flatten())
        print(label_spread.n_iter_)
        

        label_recons = np.zeros([filament_signature.shape[0],filament_signature.shape[0],filament_signature.shape[0]])
        
        for n,coords in enumerate(index):
            ix = int(coords[0])
            iy = int(coords[1])
            iz = int(coords[2])

            if label_spread.transduction_[n] == 1:
                label_recons[ix,iy,iz] = 1
            
            if label_spread.transduction_[n] == 2:
                label_recons[ix,iy,iz] = 2
        
        
        if not os.path.isdir(ref_path + 'label/spreading/'):
            os.makedirs(ref_path + 'label/spreading/')
        np.save(ref_path + 'label/spreading/' + cluster_num, label_recons)
   
        
#%%
# %%


halos_list = np.loadtxt(ref_path +  'data/' +  '300Mpc_clump_' + '01'  + '.dat')

for halo_num in range(len(halos_list)):
    
    if halos_list[halo_num,7] >= 1 and halos_list[halo_num,6] >= 2:
        cluster_list.append(halo_num)
    elif (halos_list[halo_num,7] >= 0.01 and halos_list[halo_num,7] < 1) and ( halos_list[halo_num,6] >= 0.3 and halos_list[halo_num,6] < 2) :
        group_list.append(halo_num)



#%%
np.save(ref_path + 'label/spreading/label_raw' , label)
for ix in range(int(label.shape[0]/2)-int(vr),int(label.shape[0]/2)+int(vr)+1):
    for iy in range(int(label.shape[0]/2)-int(vr),int(label.shape[0]/2)+int(vr)+1):
        for iz in range(int(label.shape[0]/2)-int(vr),int(label.shape[0]/2)+int(vr)+1):
            if np.sqrt( (ix-int(label.shape[0]/2))**2 + (iy-int(label.shape[0]/2))**2 + (iz-int(label.shape[0]/2))**2) <= 4:
                label[ix,iy,iz] = 1