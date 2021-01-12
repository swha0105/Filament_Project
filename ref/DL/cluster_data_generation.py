#%%
import numpy as np
#%%
res = 2048
box_length = '200Mpc'
box_num = '1'

q = 'xray'

cluster_grid = 200*2 + 1

ref_path = '/storage/filament/works_v4/data/200Mpc_1/DL/'

quantity = np.fromfile('/storage/filament/works_v4/data/' + box_length + '_' + box_num + '/raw/L' + box_length[:3] + q,dtype=np.float32).reshape([res,res,res])
save_path = ref_path + 'raw_data/' + q +'/'

if not os.path.isdir(save_path):
    os.makedirs(save_path)
#%%
coords = np.loadtxt('/storage/filament/works_v4/data/200Mpc_1/raw/' + 'Cinfo.dat',dtype=np.int32)[:,1:4]
cluster_list = np.linspace(1,len(coords),len(coords),dtype=np.int32)

ref_distance = int(cluster_grid/40)*10

cluster_del_candidate = []
for num in range(len(coords)):
    num = int(num)
    ref_x = coords[num-1,0]
    ref_y = coords[num-1,1]
    ref_z = coords[num-1,2]
    for n in  range(num+1,len(coords) ):
        x = coords[n-1,0]
        y = coords[n-1,1]
        z = coords[n-1,2]

        if np.sqrt(( (ref_x-x)**2 + (ref_y-y)**2 + (ref_z-z)**2 ) ) < ref_distance:
            cluster_del_candidate.append(num)
            cluster_del_candidate.append(n)
        else:
            continue

        
cluster_virgo = np.setdiff1d(cluster_list,cluster_del_candidate)   
#%%

for cluster_num in cluster_virgo:
    cluster_dens = []

    iz = coords[cluster_num,0]
    iy = coords[cluster_num,1]
    ix = coords[cluster_num,2]
    print(cluster_num)
    cluster_quantity = []
    for iiz in range(iz - int(cluster_grid/2), iz + int(cluster_grid/2) + 1):
        for iiy in range(iy - int(cluster_grid/2), iy + int(cluster_grid/2) + 1):
            for iix in range(ix - int(cluster_grid/2), ix + int(cluster_grid/2) + 1):
                
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

                cluster_quantity.append(np.log10(quantity[iix,iiy,iiz]))

    cluster_quantity = np.array(cluster_quantity).reshape([cluster_grid,cluster_grid,cluster_grid])

    np.save(save_path + str(cluster_num) ,cluster_quantity)


# %%
