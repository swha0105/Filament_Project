#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


def gaussain_fileter_3d(sigma,array):
    size = test_array.shape[0]
    g = np.zeros([size,size,size])
    for i in range(size):
        for j in range(size):
            for k in range(size):
                g[i,j,k] = (1.0/(2*np.pi*sigma**2) * np.exp(- ( (i-size/2)**2 +(j-size/2)**2 + (k-size/2)**2 )  /(2*sigma**2)) )
    return g 


# In[ ]:


def find_virgo_like_cluster()
    path = '/storage/filament/result/cluster_3d/40Mpc/density_smoothing_plot_5_v2/'
    virgo_list = np.genfromtxt(path + 'virgo_list',dtype='str')
    virgo_path = []
    for i in range(len(virgo_list)):
        if int(virgo_list[i][0]) < 10:
            tmp_path = str('box' + virgo_list[i][0] + '/subbox' + virgo_list[i][1] + '/' + virgo_list[i][2])
            virgo_path.append(path + tmp_path)
        elif    int(virgo_list[i][0]) >= 10:
            box_num = int(int(virgo_list[i][0])/10)
            tmp_path = str('box0' + str(box_num) + '/subbox' + virgo_list[i][1] + '/' + virgo_list[i][2])
            virgo_path.append(path + tmp_path)
    return virgo_path

