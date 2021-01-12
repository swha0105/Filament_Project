#%%
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import sys
import gc


#%%
ref_path = '/storage/filament/works_v6/300Mpc_1/'

cluster_save_path = ref_path + 'cluster_box/' 

vx = np.load(cluster_save_path + 'vx/' + '1.npy')
vy = np.load(cluster_save_path + 'vy/' + '1.npy')
vz = np.load(cluster_save_path + 'vz/' + '1.npy')


# %%
