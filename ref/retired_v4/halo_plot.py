#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 
import copy

# %%
box_length = '300Mpc'
box_num = '1'

ref_path = '/storage/filament/works_v5/' 

path = ref_path + box_length + '_' + box_num + '/data/'

# %%
data = np.loadtxt(path + '300Mpc_clump_0' + box_num + '.dat')

# %%
vr = data[:,4]
dens = data[:,5]
temp = data[:,6]
xray = data[:,7]


excluded = np.where(vr==4.1)

#temp ~ 175keV is excluded

#outlier = np.argmax(temp)

# print(temp[outlier],xray[outlier])
dens = np.delete(dens,excluded)
temp = np.delete(temp,excluded)
xray = np.delete(xray,excluded)
# %%
plt.plot((temp))

# %%
plt.scatter(np.log10(temp),np.log10(xray))
#plt.scatter(temp,xray)
#plt.xlim([0,50])
#plt.ylim([0,20])

# %%
plt.scatter(np.log10(temp),np.log10(dens))

#%%
plt.hist(temp)

# %%
#plt.hist(xray,bins=np.linspace(3.16433800e-06,0.1,20))
plt.hist(xray)
plt.xscale('log')


# %%
