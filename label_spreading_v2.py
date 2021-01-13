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

    ref_path = '/storage/filament/works_v7/' + box_length + '_' + str(box_num) + '/'
    vr_list = np.loadtxt(ref_path + 'cluster_box/virial_radius')
    for cluster_num in np.sort(np.array(os.listdir(ref_path + 'pyramid/xray/gaussian/2/'))):
        print(cluster_num)

        filament_signature = np.load(ref_path + 'signature/filament/' + cluster_num)
        cluster_signature = np.load(ref_path + 'signature/cluster/' + cluster_num)
        wall_signature = np.load(ref_path + 'signature/wall/' + cluster_num)
        
        xray = np.load(ref_path +  'pyramid/xray/gaussian/2/' + cluster_num)
        xray = xray[:filament_signature.shape[0],:filament_signature.shape[0],:filament_signature.shape[0]]

        temp = np.load(ref_path +  'pyramid/temp/gaussian/2/' + cluster_num)
        temp = temp[:filament_signature.shape[0],:filament_signature.shape[0],:filament_signature.shape[0]]

        log_xray_range = np.linspace(np.min(xray),np.max(xray),100)

        label_raw = np.full([filament_signature.shape[0],filament_signature.shape[0],filament_signature.shape[0]],-1)
        label = void_detect(label_raw,temp)




        vr = 3 
        for ix in range(int(label.shape[0]/2)-vr, int(label.shape[0]/2)+vr+1):
            for iy in range(int(label.shape[0]/2)-vr, int(label.shape[0]/2)+vr+1):
                for iz in range(int(label.shape[0]/2)-vr, int(label.shape[0]/2)+vr+1):
                    if np.sqrt( (ix-int(label.shape[0]/2))**2 + (iy-int(label.shape[0]/2) )**2 + (iz-int(label.shape[0]/2) )**2 ) <= int(vr):
                            label[ix,iy,iz] = 1
                        
                                    
                                    
                                #   print("h")
        

        data_flat = np.zeros([4,len(cluster_signature.flatten())])
        
        cluster_norm = (cluster_signature.flatten() - np.min(cluster_signature.flatten()) ) / ( np.max(cluster_signature.flatten()) - np.min(cluster_signature.flatten()))
        filament_norm = (filament_signature.flatten() - np.min(filament_signature.flatten()) ) / ( np.max(filament_signature.flatten()) - np.min(filament_signature.flatten()))
        wall_norm = (wall_signature.flatten() - np.min(wall_signature.flatten()) ) / ( np.max(wall_signature.flatten()) - np.min(wall_signature.flatten()))
        xray_norm = (xray.flatten()- np.min(xray.flatten() )) / ( np.max(xray.flatten()) - np.min(xray.flatten()))

        data_flat[0,:] = cluster_norm
        data_flat[1,:] = filament_norm
        data_flat[2,:] = wall_norm
        data_flat[3,:] = xray_norm
        
        label_spread = LabelSpreading(kernel='knn', alpha=0.3,n_jobs=-1,n_neighbors=7)
                
        label_spread.fit(data_flat.T, label.flatten())
        print(label_spread.n_iter_)
               
        
        if not os.path.isdir(ref_path + 'label/spreading/'):
            os.makedirs(ref_path + 'label/spreading/')
        np.save(ref_path + 'label/spreading/' + cluster_num, label_spread.transduction_.reshape(filament_signature.shape))
   
        
#%%

a = label_spread.transduction_.reshape(filament_signature.shape)

plt.imshow(a[:,:,32])
np.save('/storage/filament/works_v7/300Mpc_1/label/spreading/1.npy',a)
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