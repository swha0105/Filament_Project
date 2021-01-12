#%%
import numpy as np
import matplotlib.pyplot as plt
#from scipy.fftpack import fft, ifft, rfft, irfft
#from scipy.fft import rfftn
import copy 
from numpy.fft import rfftn,irfftn,irfft,ifft,fftn,ifftn
import time

#%%
def hessian_matrix(array):
    length = array.shape[0]
    array_1 = np.zeros([3,length-1,length-1,length-1])
    array_2 = np.zeros([9,length-2,length-2,length-2])

    # gradient of function
    for i in range(length-1):
        for j in range(length-1):
            for k in range(length-1):

                array_1[0,i,j,k] = (array[i+1,j,k] - array[i,j,k])  # df/dx
                array_1[1,i,j,k] = (array[i,j+1,k] - array[i,j,k])  # df/dy
                array_1[2,i,j,k] = (array[i,j,k+1] - array[i,j,k])  # df/dz



    for i in range(length-2):
        for j in range(length-2):
            for k in range(length-2):

                array_2[0,i,j,k] = (array_1[0,i+1,j,k] - array_1[0,i,j,k])
                array_2[1,i,j,k] = (array_1[1,i+1,j,k] - array_1[1,i,j,k])
                array_2[2,i,j,k] = (array_1[2,i+1,j,k] - array_1[2,i,j,k])

                array_2[3,i,j,k] = (array_1[0,i,j+1,k] - array_1[0,i,j,k])
                array_2[4,i,j,k] = (array_1[1,i,j+1,k] - array_1[1,i,j,k])
                array_2[5,i,j,k] = (array_1[2,i,j+1,k] - array_1[2,i,j,k])
                
                array_2[6,i,j,k] = (array_1[0,i,j,k+1] - array_1[0,i,j,k])
                array_2[7,i,j,k] = (array_1[1,i,j,k+1] - array_1[1,i,j,k])
                array_2[8,i,j,k] = (array_1[2,i,j,k+1] - array_1[2,i,j,k])

    return array_2

def step_function(input):
    if input >= 0:
        output = 1
    else:
        output = 0 

    return output


#%%
box_length = 40
distance = 201

t = np.load('/storage/filament/works_v2/data_filament/200Mpc_1_1024/clusters/1/dens.npy')
t = t.reshape([distance,distance,distance])

ref_mean = np.mean(t)
t = t[1:,1:,1:]
fft_array = fftn(np.log10(t))


#%%

#smoothing_scale_list = [0.5,0.7,1.0,1.4,2.0,2.8,4.0]
smoothing_scale_list = [4.0]
#tmp = np.zeros([distance-1,distance-1,distance-1])
signature = np.zeros([3,distance-2,distance-2,distance-2])

tmp = copy.deepcopy(fft_array)

for smoothing_scale in smoothing_scale_list:
    print(smoothing_scale)
    for ix in range(int(fft_array.shape[0]/2)):
        for iy in range(int(fft_array.shape[1]/2)):
            for iz in range(int(fft_array.shape[2]/2)):

                k = np.sqrt( (ix/40)**2 + (iy/40)**2 + (iz/40)**2  )

                tmp[ix,iy,iz] = fft_array[ix,iy,iz]*np.exp( (-(k**2)*smoothing_scale**2)/2)
#                tmp[ix,iy,iz] = fft_array[ix,iy,iz]*0.01

        for iy in range(100,200):
            for iz in range(100,200):

                k = np.sqrt( (ix/40)**2 + ((iy-200)/40)**2 + ((iz-200)/40)**2  )

                tmp[ix,iy,iz] = fft_array[ix,iy,iz]*np.exp( (-(k**2)*smoothing_scale**2)/2)
                # tmp[ix,iy,iz] = fft_array[ix,iy,iz]*0.01

        for iy in range(100,200):
            for iz in range(int(fft_array.shape[2]/2)):

                k = np.sqrt( (ix/40)**2 + ((iy-200)/40)**2 + (iz/40)**2  )

                tmp[ix,iy,iz] = fft_array[ix,iy,iz]*np.exp( (-(k**2)*smoothing_scale**2)/2)
                # tmp[ix,iy,iz] = fft_array[ix,iy,iz]*0.01


        for iy in range(int(fft_array.shape[1]/2)):
            for iz in range(100,200):

                k = np.sqrt( (ix/40)**2 + ((iy)/40)**2 + ((iz-200)/40)**2  )

                tmp[ix,iy,iz] = fft_array[ix,iy,iz]*np.exp( (-(k**2)*smoothing_scale**2)/2)
                # tmp[ix,iy,iz] = fft_array[ix,iy,iz]*0.01




    for ix in range(100,200):
        for iy in range(100,200):
            for iz in range(100,200):

                k = np.sqrt( ((ix-200)/40)**2 + ((iy-200)/40)**2 + ((iz-200)/40)**2  )

                tmp[ix,iy,iz] = fft_array[ix,iy,iz]*np.exp( (-(k**2)*smoothing_scale**2)/2)
                # tmp[ix,iy,iz] = fft_array[ix,iy,iz]*0.01

        for iy in range(int(fft_array.shape[1]/2)):
            for iz in range(100,200):

                k = np.sqrt( ((ix-200)/40)**2 + ((iy)/40)**2 + ((iz-200)/40)**2  )

                tmp[ix,iy,iz] = fft_array[ix,iy,iz]*np.exp( (-(k**2)*smoothing_scale**2)/2)
                # tmp[ix,iy,iz] = fft_array[ix,iy,iz]*0.01

        for iy in range(100,200):
            for iz in range(int(fft_array.shape[1]/2)):

                k = np.sqrt( ((ix-200)/40)**2 + ((iy-200)/40)**2 + ((iz)/40)**2  )

                tmp[ix,iy,iz] = fft_array[ix,iy,iz]*np.exp( (-(k**2)*smoothing_scale**2)/2)
                # tmp[ix,iy,iz] = fft_array[ix,iy,iz]*0.01


        for iy in range(int(fft_array.shape[1]/2)):
            for iz in range(int(fft_array.shape[1]/2)):

                k = np.sqrt( ((ix-200)/40)**2 + ((iy)/40)**2 + ((iz)/40)**2  )

                tmp[ix,iy,iz] = fft_array[ix,iy,iz]*np.exp( (-(k**2)*smoothing_scale**2)/2)            
                # tmp[ix,iy,iz] = fft_array[ix,iy,iz]*0.01



    frn = 10**ifftn(tmp)
#%%    

pic_path = '/storage/filament/works_v3/result/'

plt.figure(figsize=[10,10])
plt.imshow(np.log10(norm_frn[:,:,100]))
plt.colorbar()
plt.axis('off')
plt.title('Log density smoothing with 4Mpc',fontsize=30)
plt.savefig(pic_path + '4Mpc')

#%%

pic_path = '/storage/filament/works_v3/result/'

plt.figure(figsize=[10,10])
plt.imshow(np.log10(t[:,:,100]))
plt.colorbar()
plt.axis('off')
plt.title('Log density ',fontsize=30)
plt.savefig(pic_path + 'ref')



#%%
    norm_frn = (ref_mean/np.mean(np.sqrt(frn.real**2  + frn.imag**2 ) ) )  * ( np.sqrt(frn.real**2   +  frn.imag**2 ) ) 

    test = hessian_matrix(norm_frn)*smoothing_scale**2

    eigenvalues = np.zeros([3,distance-2,distance-2,distance-2])

    for i in range(distance-3):
        for j in range(distance-3):
            for k in range(distance-3):
                w,_ = np.linalg.eig(test[:,i,j,k].reshape([3,3]))
                w = np.sort(w)
                eigenvalues[0,i,j,k] = w[0]  #lambda 1
                eigenvalues[1,i,j,k] = w[1]  #lambda 2
                eigenvalues[2,i,j,k] = w[2]  #lambda 3


    for i in range(distance-3):
        for j in range(distance-3):
            for k in range(distance-3):
                

                signature[0,i,j,k] = np.max([signature[0,i,j,k],np.abs(eigenvalues[2,i,j,k]/eigenvalues[0,i,j,k])* \
                                np.abs(eigenvalues[2,i,j,k])*step_function(-eigenvalues[0,i,j,k])*step_function(-eigenvalues[1,i,j,k])*step_function(-eigenvalues[2,i,j,k])] )

                # signature[0,i,j,k] = np.abs(eigenvalues[2,i,j,k]/eigenvalues[0,i,j,k])* \
                #                 step_function(-eigenvalues[0,i,j,k])*step_function(-eigenvalues[1,i,j,k])*step_function(-eigenvalues[2,i,j,k])


                signature[1,i,j,k] = np.max([signature[1,i,j,k],np.abs(eigenvalues[1,i,j,k]/eigenvalues[0,i,j,k]) * (1 - np.abs(eigenvalues[2,i,j,k]/eigenvalues[0,i,j,k])) * step_function(1 - np.abs(eigenvalues[2,i,j,k]/eigenvalues[0,i,j,k]))* \
                                np.abs(eigenvalues[1,i,j,k])*step_function(-eigenvalues[0,i,j,k])*step_function(-eigenvalues[1,i,j,k])])

                # signature[1,i,j,k] = np.abs(eigenvalues[1,i,j,k]/eigenvalues[0,i,j,k]) * (1 - np.abs(eigenvalues[2,i,j,k]/eigenvalues[0,i,j,k])) * step_function(1 - np.abs(eigenvalues[2,i,j,k]/eigenvalues[0,i,j,k]))* \
                #                 step_function(-eigenvalues[0,i,j,k])*step_function(-eigenvalues[1,i,j,k])


                signature[2,i,j,k] = np.max([signature[2,i,j,k],(1-np.abs(eigenvalues[1,i,j,k]/eigenvalues[0,i,j,k]))*step_function(1-np.abs(eigenvalues[1,i,j,k]/eigenvalues[0,i,j,k])) *\
                                (1-np.abs(eigenvalues[2,i,j,k]/eigenvalues[0,i,j,k]))*step_function(1-np.abs(eigenvalues[2,i,j,k]/eigenvalues[0,i,j,k])) *\
                                np.abs(eigenvalues[0,i,j,k])*step_function(-eigenvalues[0,i,j,k])] )

                # signature[2,i,j,k] = (1-np.abs(eigenvalues[1,i,j,k]/eigenvalues[0,i,j,k]))*step_function(1-np.abs(eigenvalues[1,i,j,k]/eigenvalues[0,i,j,k])) *\
                #                 (1-np.abs(eigenvalues[2,i,j,k]/eigenvalues[0,i,j,k]))*step_function(1-np.abs(eigenvalues[2,i,j,k]/eigenvalues[0,i,j,k])) *\
                #                 step_function(-eigenvalues[0,i,j,k])

#%%


pic_path = '/storage/filament/works_v3/result/'

plt.figure(figsize=[10,10])
plt.imshow(np.log10(signature[2,:,:,100]+0.01))
#plt.colorbar()
plt.axis('off')
plt.title('wall signature ',fontsize=30)
plt.savefig(pic_path + 'wall')



#%%
np.savetxt('/storage/filament/works_v3/nexus/200Mpc_1_1024/non-smoothing/1',signature.reshape([3,-1]))
#%%
cluster = signature[0,:,:,:]
filament = signature[1,:,:,:]
wall = signature[2,:,:,:]
#%%
np.savetxt('/storage/filament/works_v3/nexus/200Mpc_1_1024/non-smoothing/1_cluster',cluster.flatten())
np.savetxt('/storage/filament/works_v3/nexus/200Mpc_1_1024/non-smoothing/1_filament',filament.flatten())
np.savetxt('/storage/filament/works_v3/nexus/200Mpc_1_1024/non-smoothing/1_wall',wall.flatten())
#%%


#%%

bins = np.linspace(-1,np.log10(np.max(filament)),50)

filament_mass_list = []
wall_mass_list = []

refined = np.zeros([199,199,199])

for threshold in bins:
    filament_mass = 0 
    wall_mass = 0 
    #print(10**threshold)
    for iz in range(signature.shape[1]):
        for iy in range(signature.shape[1]):
            for ix in range(signature.shape[1]):
                if signature[1,ix,iy,iz] >= 10**threshold:
                    filament_mass = filament_mass + t[ix,iy,iz]
                if signature[2,ix,iy,iz] >= 10**threshold :
                    wall_mass = wall_mass + t[ix,iy,iz]                    
                    
                if signature[1,ix,iy,iz] >= 10**-0.86 and signature[2,ix,iy,iz] < 10**0.06:
                    refined[ix,iy,iz] = signature[1,ix,iy,iz]


    filament_mass_list.append(filament_mass)
    wall_mass_list.append(wall_mass)
                # if signature[2,ix,iy,iz] >= 10**threshold:
                #     wall_mass = wall_mass + t[ix,iy,iz]
 
    #print(np.abs(np.log10(filament_mass) / threshold),10**threshold )

    #filament_mass_list.append(np.abs(np.log10(filament_mass) / threshold))
#    wall_mass = np.abs(np.log10(wall_mass) / np.log10(threshold))
    #filament_mass_diff.append(filament_mass)
#    wall_mass_diff.append(wall_mass)
    

#%%
filament_mass_list_2 = []
wall_mass_list_2 = []

for i in range(len(filament_mass_list)-1):
    filament_mass_list_2.append(np.abs(filament_mass_list[i]**2 - filament_mass_list[i+1]**2))
    wall_mass_list_2.append(np.abs(wall_mass_list[i]**2 - wall_mass_list[i+1]**2))


#%%
plt.figure(figsize=[8,8])
plt.plot(bins[1:],filament_mass_list_2,'r')
plt.plot(bins[1:],wall_mass_list_2,'b')
plt.savefig('/storage/signature.png')
#plt.plot(bins,(wall_mass_diff),'b')

#%%
i = 100
plt.figure(figsize=[16,8])
plt.subplot(121)
plt.imshow(refined[:,:,i])
plt.colorbar()
plt.subplot(122)
#plt.imshow(np.log10(t[:,:,100]))
plt.imshow(signature[1,:,:,i])
plt.clim([0, 2.5])
plt.colorbar()

#%

np.savetxt('/storage/filament/works_v3/nexus/filament_refiend',refined.flatten())


#%%

signature_4 = np.zeros([3,distance-3,distance-3,distance-3])
for i in range(distance-3):
    for j in range(distance-3):
        for k in range(distance-3):
            

            signature_4[0,i,j,k] = np.abs(eigenvalues[2,i,j,k]/eigenvalues[0,i,j,k])* \
                            np.abs(eigenvalues[2,i,j,k])*step_function(-eigenvalues[0,i,j,k])*step_function(-eigenvalues[1,i,j,k])*step_function(-eigenvalues[2,i,j,k])

            # signature[0,i,j,k] = np.abs(eigenvalues[2,i,j,k]/eigenvalues[0,i,j,k])* \
            #                 step_function(-eigenvalues[0,i,j,k])*step_function(-eigenvalues[1,i,j,k])*step_function(-eigenvalues[2,i,j,k])


            signature_4[1,i,j,k] = np.abs(eigenvalues[1,i,j,k]/eigenvalues[0,i,j,k]) * (1 - np.abs(eigenvalues[2,i,j,k]/eigenvalues[0,i,j,k])) * step_function(1 - np.abs(eigenvalues[2,i,j,k]/eigenvalues[0,i,j,k]))* \
                            np.abs(eigenvalues[1,i,j,k])*step_function(-eigenvalues[0,i,j,k])*step_function(-eigenvalues[1,i,j,k])

            # signature[1,i,j,k] = np.abs(eigenvalues[1,i,j,k]/eigenvalues[0,i,j,k]) * (1 - np.abs(eigenvalues[2,i,j,k]/eigenvalues[0,i,j,k])) * step_function(1 - np.abs(eigenvalues[2,i,j,k]/eigenvalues[0,i,j,k]))* \
            #                 step_function(-eigenvalues[0,i,j,k])*step_function(-eigenvalues[1,i,j,k])


            signature_4[2,i,j,k] = (1-np.abs(eigenvalues[1,i,j,k]/eigenvalues[0,i,j,k]))*step_function(1-np.abs(eigenvalues[1,i,j,k]/eigenvalues[0,i,j,k])) *\
                            (1-np.abs(eigenvalues[2,i,j,k]/eigenvalues[0,i,j,k]))*step_function(1-np.abs(eigenvalues[2,i,j,k]/eigenvalues[0,i,j,k])) *\
                            np.abs(eigenvalues[0,i,j,k])*step_function(-eigenvalues[0,i,j,k])

            # signature[2,i,j,k] = (1-np.abs(eigenvalues[1,i,j,k]/eigenvalues[0,i,j,k]))*step_function(1-np.abs(eigenvalues[1,i,j,k]/eigenvalues[0,i,j,k])) *\
            #                 (1-np.abs(eigenvalues[2,i,j,k]/eigenvalues[0,i,j,k]))*step_function(1-np.abs(eigenvalues[2,i,j,k]/eigenvalues[0,i,j,k])) *\
            #                 step_function(-eigenvalues[0,i,j,k])