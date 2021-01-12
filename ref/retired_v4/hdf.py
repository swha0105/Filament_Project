#%%
import h5py 
import numpy as np

ref_path = '/storage/Codes/git/sibal/indoor3d_ins_seg_hdf5/''

# %%
f = h5py.File(ref_path + 'Area_1_WC_1.h5', 'r')


# %%
coords = f['coords'].value
labels = f['labels'].value
points = f['points'].value

# %%
