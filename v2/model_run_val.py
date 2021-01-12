#%%
import numpy as np
import sys 
import os
import matplotlib.pyplot as pt
#from utils import *
import copy
#import models
import cv2
#from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
#import skimage.io as io
#import skimage.transform as trans
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler 
from keras import backend as keras
import keras.backend.tensorflow_backend as K
import matplotlib.pyplot as plt
from keras.utils import plot_model
from contextlib import redirect_stdout
#from unet_models import *

#box_length = '300Mpc'

box_length_list = ['300Mpc']
box_num = ['1','2']

dens_path_list = []
label_path_list = []


for box_length in box_length_list:
    for box_n in box_num:
        ref_path = '/storage/filament/works_v4/data/' + box_length + '_' + box_n + '/DL/'

        tmp_path = ref_path + 'smoothing/'

        for tmp_name in os.listdir(tmp_path + 'label/'):
            
            dens_path_list.append(tmp_path +  'dens/' + tmp_name)
            dens_path_list.append(tmp_path +  'augmented/dens/' + tmp_name)
            dens_path_list.append(tmp_path +  'augmented_2/dens/' + tmp_name)
            label_path_list.append(tmp_path +  'label/' + tmp_name)
            label_path_list.append(tmp_path +  'augmented/label/' + tmp_name)
            label_path_list.append(tmp_path +  'augmented_2/label/' + tmp_name)
    


array_size = 73
number_of_data = len(dens_path_list)
dens_list = np.zeros([number_of_data,array_size-1,array_size-1,array_size-1])
label_list = np.zeros([number_of_data,array_size-1,array_size-1,array_size-1])

validation_num = np.random.randint(number_of_data,size=int(0.2*number_of_data))

data_list = []

for n,dens_num_path in enumerate(dens_path_list):
    dens_list[n,...] = np.load(dens_num_path)[:array_size-1,:array_size-1,:array_size-1]
    data_list.append(dens_num_path)
for n,label_num_path in enumerate(label_path_list):
    label_list[n,...] = np.load(label_num_path)[:array_size-1,:array_size-1,:array_size-1]
    

dens_train = np.zeros([number_of_data-len(validation_num),array_size-1,array_size-1,array_size-1])
label_train = np.zeros([number_of_data-len(validation_num),array_size-1,array_size-1,array_size-1])

dens_val = np.zeros([len(validation_num),array_size-1,array_size-1,array_size-1])
label_val = np.zeros([len(validation_num),array_size-1,array_size-1,array_size-1])

n = 0
m = 0

val_list = []

for num in range(dens_list.shape[0]):
    if num in validation_num:
        dens_val[n,...] = dens_list[num,...]
        label_val[n,...] = label_list[num,...]
        val_list.append(dens_path_list[num])
        n = n +1
    else:
        dens_train[n,...] = dens_list[num,...]
        label_train[n,...] = label_list[num,...]
        m = m + 1

label_train_ex = np.expand_dims(label_train,axis=-1)
dens_train_ex = np.expand_dims(dens_train,axis=-1)

label_val_ex = np.expand_dims(label_val,axis=-1)
dens_val_ex = np.expand_dims(dens_val,axis=-1)

learning_rate = 1e-3
decay_rate = learning_rate/300


def dice_coef(y_true, y_pred):
    smooth = 1e-7
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth * 0.01) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


#%%
run_name = '0618'
model_name = 'unet'

#model name = unet, resnet
sys.path.append('/storage/Codes/Research/Filaments/models/')
n = 8 #(number of filter)
m = 3 #(convolution size)
s = 2 #(pooling size)


if model_name == 'unet':
   from unet_models import *
   model = get_unet_v1(n,m,s)

if model_name == 'densenet':
    from DenseNet3D import *
    model = DenseNet3D_FCN((72, 72, 72, 1), nb_dense_block=2, growth_rate=2,
            nb_layers_per_block=2, upsampling_type='upsampling', classes=1, activation='sigmoid')
    model.compile(optimizer=Adam(lr=learning_rate, decay=decay_rate), loss=dice_coef_loss, metrics=[dice_coef])

run_path = '/storage/filament/works_v4/results/'  + run_name + '/'

if not os.path.isdir(run_path):
    os.makedirs(run_path)
os.chdir(run_path)

check_name =  'checkpoint_' + run_name + '{epoch:02d}-{val_loss:.4f}_.hdf5'
cb_checkpoint = ModelCheckpoint(run_path + check_name, monitor='val_dice_coef', verbose=1, mode='max', save_best_only=True, save_weights_only=False, period=1)
#%%
#plot_model
#plot_model(model, to_file='model.png')
os.chdir(run_path)

with open('modelsummary.txt', 'w') as f:
    with redirect_stdout(f):
        #if model_name=='unet':
            #print()
        model.summary()



#%%
#
hist = model.fit(dens_train_ex, label_train_ex, batch_size=8, epochs=300,shuffle=True, callbacks = [cb_checkpoint], validation_data=(dens_val_ex, label_val_ex))

#%%

plt.figure(figsize=[10,10])
plt.plot(np.abs(hist.history['dice_coef']),label='dice_coef')
plt.plot(np.abs(hist.history['val_dice_coef']),label='val_dice_coef')
plt.legend(fontsize=20)
plt.xlabel('Epochs',fontsize=20)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
#plt.ylabel('',fontsize=20)

plt.title(model_name,fontsize=20)
plt.savefig(run_path + 'dice_coef.png')

model.save_weights(run_path + 'model_' + run_name + ' n=' + str(n) + ' m=' + str(m) + ' s=' + str(s) + '.h5' )
np.savetxt(run_path + 'validation_set.txt',np.array(val_list),fmt='%s')


#%%

#tmp = np.expand_dims(dens_list,axis=0)
#test_dens = np.expand_dims(dens_list,axis=-1)
if not os.path.isdir(run_path + 'predict/'):
    os.makedirs(run_path + 'predict/')
    
for i in range(len(dens_val_ex)):
    test = model.predict(dens_val_ex[i:i+1])
    np.save(run_path + 'predict/' + str(i), test[0,:,:,:,0])

#%%
# i = 61
# j = 23
# plt.figure(figsize=[16,8])
# plt.subplot(121)
# plt.contourf(img_val[i,j,:,:,0])
# plt.subplot(122)
# plt.contourf(label_val[i,j,:,:,0])
# plt.gray()


# # In[70]:


# #img_val = img_file[:16,:,:,:,:]
# save_dir = '/storage/filament/result/cluster_3d/40Mpc/prediction/aug_xyz_shuffle/'
# # for i in range(test_re.shape[0]):
# #     if not os.path.isdir(save_dir + str(i) +'/'):
# #         os.makedirs(save_dir + str(i) +'/')
# #     os.chdir(save_dir + str(i) +'/')
# for j in range(test_re.shape[1]):
#         plt.figure(figsize=[30,30])
#         plt.imsave(str(j) +'.png',test_re[i,j,:,:],cmap='gray',dpi=300,format='png')


# # In[ ]:


# model.load_weights('/home/swha/filament/models/unet/0212/checkpoint_021292--0.4909_.hdf5')


# # In[10]:


# val = model.predict(img_val)
# #%%

# label_val_re = label_val.reshape([24,96,96,96])
# val_re = val.reshape([24,96,96,96])
# #%%
# i = 2
# j = 60
# # plt.figure(figsize=[10,10])
# # plt.imshow(img_val[i,:,:,0])
# # plt.show()
# plt.figure(figsize=[16,8])
# plt.subplot(121)
# plt.imshow(label_val_re[i,j,:,:])
# plt.gray()
# plt.subplot(122)
# plt.imshow(val_re[i,j,:,:])
# plt.gray()
# plt.show()
# # In[55]:

# #%%
# predict_path = '/home/swha/filament/data/prediction/0213/'

# os.chdir(predict_path + 'label/')
# j =  12
# for i in range(96):
#     plt.imsave(str(i) + '.png',label_val_re[j,i,:,:],cmap='gray',dpi=300,format='png')
# os.chdir(predict_path + 'predict/')
# for i in range(96):
#     plt.imsave(str(i) + '.png',val_re[j,i,:,:],cmap='gray',dpi=300,format='png')
    
# #%%
# os.chdir('/storage/filament/result/cluster_3d/40Mpc/prediction/aug_xyz/pic/')
# for i in range(16):
#     for j in range(24):
#         plt.figure(figsize=[30,30])
#         plt.imsave('i_' + str(i) + ' j_' + str(j) +'.png',test[i,j,:,:,0],cmap='gray',dpi=300,format='png')


# # In[ ]:



