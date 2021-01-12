#%%
import numpy as np
import matplotlib.pyplot as plt
import os 
import copy
import sys

def filp_augmentation(input,random_int_x,random_int_y,random_int_z):
    
    output = np.zeros([input.shape[0],input.shape[0],input.shape[0]])

    if random_int_x == 0:
        ix = 0
    if random_int_x == 1:
        ix = input.shape[0]-1
    if random_int_y == 0:
        iy = 0
    if random_int_y == 1:
        iy = input.shape[0]-1
    if random_int_z == 0:
        iz = 0
    if random_int_z == 1:
        iz = input.shape[0]-1


    for i in range(input.shape[0]):
        for j in range(input.shape[0]):
            for k in range(input.shape[0]):
    
                output[i,j,k] = input[np.abs(i-ix),np.abs(j-iy),np.abs(k-iz)]


    return output
# %%

box_length = '300Mpc'
cluster_length = 45
box_list = ['1','2']
#box_num = '1'

for box_num in box_list:
    ref_path = '/storage/filament/works_v4/data/' + box_length + '_' + str(box_num) + '/DL/'

    label_path = ref_path + 'smoothing/label/'
    dens_path = ref_path + 'smoothing/dens/'

    label_save_path = ref_path + 'smoothing/augmented/label/'
    dens_save_path = ref_path + 'smoothing/augmented/dens/'

    if not os.path.isdir(label_save_path):
        os.makedirs(label_save_path)

    if not os.path.isdir(dens_save_path):
        os.makedirs(dens_save_path)

    label_save_path_2 = ref_path + 'smoothing/augmented_2/label/'
    dens_save_path_2 = ref_path + 'smoothing/augmented_2/dens/'

    if not os.path.isdir(label_save_path_2):
        os.makedirs(label_save_path_2)
    if not os.path.isdir(dens_save_path_2):
        os.makedirs(dens_save_path_2)


    for label_num in np.sort(os.listdir(label_path)):
        print(label_num)
        label = np.load(label_path + label_num)
        dens = np.load(dens_path + label_num)[:label.shape[0],:label.shape[0],:label.shape[0]]
        
        r_x = np.random.randint(2)
        r_y = np.random.randint(2)
        r_z = np.random.randint(2)
        
        while r_x == 0 and r_y == 0 and r_z == 0:
            r_x = np.random.randint(2)
            r_y = np.random.randint(2)
            r_z = np.random.randint(2)
        


        r_x_2 = np.random.randint(2)
        r_y_2 = np.random.randint(2)
        r_z_2 = np.random.randint(2)
        
        while (r_x_2 == 0 and r_y_2 == 0 and r_z_2 == 0) or (r_x_2 == r_x and r_y_2 == r_y and r_z_2 == r_z):
            r_x_2 = np.random.randint(2)
            r_y_2 = np.random.randint(2)
            r_z_2 = np.random.randint(2)
     

        filp_dens = filp_augmentation(dens,r_x,r_y,r_z)
        filp_label = filp_augmentation(label,r_x,r_y,r_z)

        np.save(dens_save_path + label_num,filp_dens)
        np.save(label_save_path + label_num,filp_label)
        
        filp_dens_2 = filp_augmentation(dens,r_x_2,r_y_2,r_z_2)
        filp_label_2 = filp_augmentation(label,r_x_2,r_y_2,r_z_2)

        np.save(dens_save_path_2 + label_num,filp_dens_2)
        np.save(label_save_path_2 + label_num,filp_label_2)
    


    
    
#%%
i=35
plt.figure(figsize=[16,8])
plt.subplot(121)
plt.imshow(filp_dens[:,:,i])
plt.subplot(122)
plt.imshow(filp_label[:,:,i])
#%%

i=35
plt.figure(figsize=[16,8])
plt.subplot(121)
plt.imshow(dens[:,:,i])
plt.subplot(122)
plt.imshow(label[:,:,i])