
# %%
import numpy as np
import matplotlib.pyplot as plt

box_length = '300Mpc'
box_num = '1'
mpc_grid = int(box_length[:3])/2048


ref_path = '/storage/filament/works_v7/'+ box_length + '_' + box_num + '/filament/'

curvature_max = []
curvature_mean = []
filament_length = []
dens_list = []
total_num = 0
for cluster_num in np.sort(os.listdir(ref_path + 'fitting/')):
    
    cluster_path = ref_path + 'fitting/' + str(cluster_num) +'/'
    # dens =  10**np.loadtxt(ref_path + 'whole/' +  str(cluster_num) + '/' + 'dens_list' )
    # dens = np.expand_dims(dens,axis=1)
    # volume =  np.loadtxt(ref_path + 'whole/' +  str(cluster_num) + '/' + 'volume_list' )
    # volume = np.expand_dims(volume,axis=1)
    tmp_num = 0
    for filament_num in np.sort(os.listdir(cluster_path)):
        
        
        filament = np.loadtxt(cluster_path + str(filament_num))
        
        if len(filament) <= 5:
            continue
        else:
            filament_curvature = np.loadtxt(ref_path + 'curvature/' +  str(cluster_num) + '/'+ str(filament_num) )

            #filament_width =   len(np.loadtxt(ref_path + 'whole/' +  str(cluster_num) + '/'+ str(filament_num) + '_width' ))
            #filament_volume =   len(np.loadtxt(ref_path + 'whole/' +  str(cluster_num) + '/'+ str(filament_num) ))
            
            
            if np.max(filament_curvature) > 0.5:
                print(cluster_num,filament_num,len(filament))
            curvature_max.append(np.max(filament_curvature))
            curvature_mean.append(np.mean(filament_curvature))
            filament_length.append(len(filament)*mpc_grid)

            #dens_list.append( np.log10(dens[tmp_num]*volume[tmp_num] * (0.15*3*10**24)**3/(2*10**33) ) )
            total_num = total_num + 1
            tmp_num = tmp_num + 1

    
#%%
plt.figure(figsize=[10,10])
plt.scatter(dens_list,filament_length)
plt.xlabel('mass ',fontsize=20)
plt.ylabel('filament_length (Mpch^-1)',fontsize=20)


#%%
tmp_range = np.linspace(0,2,30)
plt.figure(figsize=[10,10])
plt.hist(curvature_max,tmp_range)
plt.title('max_curvature',fontsize=20)
#plt.savefig(ref_path + 'statistic/' + 'curvature_max')

plt.figure(figsize=[10,10])
plt.hist(curvature_mean)
plt.title('mean_curvature',fontsize=20)
#plt.savefig(ref_path + 'statistic/' + 'curvature_mean')

plt.figure(figsize=[10,10])
plt.scatter(filament_length,curvature_max)
plt.xlabel('filament_length (Mpch^-1)',fontsize=20)
plt.ylabel('max_curvature',fontsize=20)
#plt.savefig(ref_path + 'statistic/' + 'length_curvature_max')


# %%


save_path = '/storage/filament/works_v2/statistics/'
plt.hist(curv_mean,bins = 20, edgecolor='black',linewidth=1.2)
plt.xlabel("mean_curvature (1/Mpc)",fontsize=15)
plt.ylabel("Number of Filament",fontsize=15)
ticks = np.linspace(0,5,11)
plt.xticks(ticks)
plt.savefig(save_path + '4Mpc/' + 'mean_curvature (4Mpc)') 

#%%

ref_path = '/storage/filament/works_v7/'+ box_length + '_' + box_num + '/filament/'

total_num = 0

for cluster_num in np.sort(os.listdir(ref_path + 'whole/')):
    cluster_path = ref_path + 'whole/' + str(cluster_num) +'/'
    volume_list = []

    for filament_num in np.sort(os.listdir(cluster_path)):
        print(cluster_num,filament_num)
        if len(filament_num) <= 2:
            filament = np.loadtxt(cluster_path + str(filament_num))
            break
            if len(filament) < 5:
                continue
            else:
                volume_list.append(len(filament)) 
        else:
            continue
    

    np.savetxt(cluster_path + 'volume_list',volume_list)    


    
    

    
#%%

with open("")