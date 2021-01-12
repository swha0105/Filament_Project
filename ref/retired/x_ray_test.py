#%%
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2 

#%%
data_path = '/storage/filament/data/200Mpc_v2/box02/'
volume_const  = 0.01*( (200/2048)*(3.086/0.7))**3

xray = np.zeros(256**3,dtype='float64')
xray = np.genfromtxt(data_path + '256den18g+1024+2',dtype='float64')
xray = xray.reshape([256,256,256])
xray = xray*volume_const

# %%
nx=256
chop = 10

candidate_coordx = []
candidate_coordy = []
candidate_coordz = []
log_xray = np.log10(xray)
for iz in range(chop,nx-chop):
    for iy in range(chop,nx-chop):
        for ix in range(chop,nx-chop):
            if log_xray[ix,iy,iz] >= -5.5:
                
                radius = 0
                
                for vr in range(10):
                    critical_xray = np.mean(log_xray[ix-vr:ix+vr , iy-vr:iy+vr , iz-vr:iz+vr])
                    if critical_xray < -6:
                        break
                    else:
                        pass


                # 주변 local maximun tag
                if vr >= 2:
                    ref_value = log_xray[ix,iy,iz]
                    tmp_x = ix
                    tmp_y = iy
                    tmp_z = iz
                    max_value = ref_value
                    for rx in range(-vr,vr+1):
                        for ry in range(-vr,vr+1):
                            for rz in range(-vr,vr+1):
                                if ref_value < log_xray[ix+rx, iy+ry, iz+rz]:
                                    tmp_x = 0
                                    tmp_y = 0
                                    tmp_z = 0
                                    
                                    
                                else:
                                    pass
                                
                candidate_coordx.append(tmp_x)
                candidate_coordy.append(tmp_y)
                candidate_coordz.append(tmp_z)

# %%
coords = list(set(list(zip(candidate_coordx,candidate_coordy,candidate_coordz))))
coords = np.array(sorted(coords, key = lambda x:x[2]))

# %%
i=141

plt.figure(figsize=[10,10])
min_coords = np.min(np.where(coords[:,2]==i))
max_coords = np.max(np.where(coords[:,2]==i))
plt.contourf(log_xray[:,:,i])
plt.scatter(coords[min_coords:max_coords+1,1],coords[min_coords:max_coords+1,0],c='red')
    

# %%
