
# %%
import numpy as np
import matplotlib.pyplot as plt

box_length = '300Mpc'
box_num = '1'
mpc_grid = 4*int(box_length[:3])/2048


ref_path = '/storage/filament/works_v5/'+ box_length + '_' + box_num + '/clusters/filament/'

curvature_max = []
curvature_mean = []
filament_length = []
total_num = 0
for cluster_num in np.sort(os.listdir(ref_path + 'curvature/')):
    
    cluster_path = ref_path + 'curvature/' + str(cluster_num) +'/'
    
    for filament_num in np.sort(os.listdir(cluster_path)):
        
        filament_curvature = np.loadtxt(cluster_path + str(filament_num))
        
        curvature_max.append(np.max(filament_curvature))
        curvature_mean.append(np.mean(filament_curvature))
        filament_length.append(len(filament_curvature)*mpc_grid + 4)
        if np.max(filament_curvature) >= 0.4:
            print(cluster_num,filament_num,np.max(filament_curvature),len(filament_curvature)*mpc_grid + 4)
        total_num = total_num + 1

tmp_range = np.linspace(0,2,30)
plt.figure(figsize=[10,10])
plt.hist(curvature_max,tmp_range)
plt.title('max_curvature',fontsize=20)
plt.savefig(ref_path + 'statistic/' + 'curvature_max')

plt.figure(figsize=[10,10])
plt.hist(curvature_mean)
plt.title('mean_curvature',fontsize=20)
plt.savefig(ref_path + 'statistic/' + 'curvature_mean')

plt.figure(figsize=[10,10])
plt.scatter(filament_length,curvature_max)
plt.xlabel('filament_length (Mpch^-1)',fontsize=20)
plt.ylabel('max_curvature',fontsize=20)
plt.savefig(ref_path + 'statistic/' + 'length_curvature_max')


# %%


save_path = '/storage/filament/works_v2/statistics/'
plt.hist(curv_mean,bins = 20, edgecolor='black',linewidth=1.2)
plt.xlabel("mean_curvature (1/Mpc)",fontsize=15)
plt.ylabel("Number of Filament",fontsize=15)
ticks = np.linspace(0,5,11)
plt.xticks(ticks)
plt.savefig(save_path + '4Mpc/' + 'mean_curvature (4Mpc)') 
