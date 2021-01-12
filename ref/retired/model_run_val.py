#!/usr/bin/env python
# coding: utf-8

# In[3]:


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

# In[5]:


img_width = 96
img_height = 96
img_depth = 24

box_list = ['box01/','box02/']
aug_list = [0,90,180,270]
#aug_list = [0,180]

label_dir_pre = '/home/swha/filament/data/label_smoothing/'
img_dir_pre = '/home/swha/filament/data/density_smoothing_plot_5_v2/'
tmp_img = []
tmp_label = []


val_box_num = 'box01/'
val_sub_num = 'subbox01'
val_peak_num = '0'

val_label_dir = label_dir_pre + val_box_num + val_sub_num +'/' + val_peak_num + '/augmented/'
val_img_dir = img_dir_pre + val_box_num + val_sub_num + '/' + val_peak_num + '/'

val_box_num_1 = 'box01/'
val_sub_num_1 = 'subbox03'
val_peak_num_1 = '0'

val_label_dir_1 = label_dir_pre + val_box_num_1 + val_sub_num_1 +'/' + val_peak_num_1 + '/augmented/'
val_img_dir_1 = img_dir_pre + val_box_num_1 + val_sub_num_1 + '/' + val_peak_num_1 + '/'


val_box_num_2 = 'box02/'
val_sub_num_2 = 'subbox02'
val_peak_num_2 = '0'

val_label_dir_2 = label_dir_pre + val_box_num_2 + val_sub_num_2 +'/' + val_peak_num_2 + '/augmented/'
val_img_dir_2 = img_dir_pre + val_box_num_2 + val_sub_num_2 + '/' + val_peak_num_2 + '/'




val_label_tmp = []
val_img_tmp = []

train_data_list = []
val_data_list = []

#%%

for box_num in box_list:
    for sub_num in os.listdir(label_dir_pre + box_num):
        peak_list = os.listdir(label_dir_pre + box_num + sub_num + '/')
        for peak_num in peak_list:
            label_dir = label_dir_pre + box_num + sub_num + '/' + peak_num + '/augmented/'
            img_dir = img_dir_pre + box_num + sub_num + '/' + peak_num + '/'

            # validation 이면 따로 뽑기
            #print(box_num,sub_num,peak_num)
            if(box_num in val_box_num and sub_num in val_sub_num and peak_num in val_peak_num):
      
                for aug_dir in aug_list:
                    os.chdir(val_label_dir+str(aug_dir)+'/')
                    for nn,_ in enumerate(os.listdir(val_label_dir + str(aug_dir) + '/')):
                        val_label_tmp.append(cv2.resize(cv2.imread(val_label_dir + str(aug_dir) + '/' + str(nn) + '.png',cv2.IMREAD_GRAYSCALE),(img_width,img_height)))
                        
                    val_data_name = str("box_num " + str(box_num) + " sub_box " + str(sub_num) + " peak_num " + str(peak_num) + " aug_dir " + str(aug_dir))
                    val_data_list.append(val_data_name)
                    
                    for nn,_ in enumerate(os.listdir(val_label_dir + 'z/'+ str(aug_dir) + '/')):
                        val_label_tmp.append(cv2.resize(cv2.imread(val_label_dir + 'z/'+ str(aug_dir) + '/'+ str(nn) + '.png',cv2.IMREAD_GRAYSCALE),(img_width,img_height)))
      
                    val_data_name = str("box_num " + str(box_num) + " sub_box " + str(sub_num) + " peak_num " + str(peak_num) + " aug_dir " + str(aug_dir) + 'with z')
                    val_data_list.append(val_data_name)
                    

                    os.chdir(val_img_dir+str(aug_dir)+'/')
                    for nn,_ in enumerate(os.listdir(val_img_dir + str(aug_dir) + '/')):
                        val_img_tmp.append(cv2.resize(cv2.imread(val_img_dir + str(aug_dir) + '/' + str(nn) + '.png',cv2.IMREAD_GRAYSCALE),(img_width,img_height)))
                    for nn,_ in enumerate(os.listdir(val_img_dir + 'z/'+ str(aug_dir) + '/')):
                        val_img_tmp.append(cv2.resize(cv2.imread(val_img_dir + 'z/'+ str(aug_dir) + '/' + str(nn) + '.png',cv2.IMREAD_GRAYSCALE),(img_width,img_height)))
            
            
            elif(box_num in val_box_num_1 and sub_num in val_sub_num_1 and peak_num in val_peak_num_1):

                for aug_dir in aug_list:
                    os.chdir(val_label_dir_1+str(aug_dir)+'/')
                    for nn,_ in enumerate(os.listdir(val_label_dir_1 + str(aug_dir) + '/')):
                        val_label_tmp.append(cv2.resize(cv2.imread(val_label_dir_1 + str(aug_dir) + '/' + str(nn) + '.png',cv2.IMREAD_GRAYSCALE),(img_width,img_height)))
                        
                    val_data_name = str("box_num " + str(box_num) + " sub_box " + str(sub_num) + " peak_num " + str(peak_num) + " aug_dir " + str(aug_dir))
                    val_data_list.append(val_data_name)
                    
                    for nn,_ in enumerate(os.listdir(val_label_dir_1 + 'z/'+ str(aug_dir) + '/')):
                        val_label_tmp.append(cv2.resize(cv2.imread(val_label_dir_1 + 'z/'+ str(aug_dir) + '/'+ str(nn) + '.png',cv2.IMREAD_GRAYSCALE),(img_width,img_height)))
      
                    val_data_name = str("box_num " + str(box_num) + " sub_box " + str(sub_num) + " peak_num " + str(peak_num) + " aug_dir " + str(aug_dir) + 'with z')
                    val_data_list.append(val_data_name)
                    

                    os.chdir(val_img_dir_1+str(aug_dir)+'/')
                    for nn,_ in enumerate(os.listdir(val_img_dir_1 + str(aug_dir) + '/')):
                        val_img_tmp.append(cv2.resize(cv2.imread(val_img_dir_1 + str(aug_dir) + '/' + str(nn) + '.png',cv2.IMREAD_GRAYSCALE),(img_width,img_height)))
                    for nn,_ in enumerate(os.listdir(val_img_dir_1 + 'z/'+ str(aug_dir) + '/')):
                        val_img_tmp.append(cv2.resize(cv2.imread(val_img_dir_1 + 'z/'+ str(aug_dir) + '/' + str(nn) + '.png',cv2.IMREAD_GRAYSCALE),(img_width,img_height)))
            
            
            
            elif(box_num in val_box_num_2 and sub_num in val_sub_num_2 and peak_num in val_peak_num_2):

                for aug_dir in aug_list:
                    os.chdir(val_label_dir_2+str(aug_dir)+'/')
                    for nn,_ in enumerate(os.listdir(val_label_dir_2 + str(aug_dir) + '/')):
                        val_label_tmp.append(cv2.resize(cv2.imread(val_label_dir_2 + str(aug_dir) + '/' + str(nn) + '.png',cv2.IMREAD_GRAYSCALE),(img_width,img_height)))
                        
                    val_data_name = str("box_num " + str(box_num) + " sub_box " + str(sub_num) + " peak_num " + str(peak_num) + " aug_dir " + str(aug_dir))
                    val_data_list.append(val_data_name)
                    
                    for nn,_ in enumerate(os.listdir(val_label_dir_2 + 'z/'+ str(aug_dir) + '/')):
                        val_label_tmp.append(cv2.resize(cv2.imread(val_label_dir_2 + 'z/'+ str(aug_dir) + '/'+ str(nn) + '.png',cv2.IMREAD_GRAYSCALE),(img_width,img_height)))
      
                    val_data_name = str("box_num " + str(box_num) + " sub_box " + str(sub_num) + " peak_num " + str(peak_num) + " aug_dir " + str(aug_dir) + 'with z')
                    val_data_list.append(val_data_name)
                    

                    os.chdir(val_img_dir_2+str(aug_dir)+'/')
                    for nn,_ in enumerate(os.listdir(val_img_dir_2 + str(aug_dir) + '/')):
                        val_img_tmp.append(cv2.resize(cv2.imread(val_img_dir_2 + str(aug_dir) + '/' + str(nn) + '.png',cv2.IMREAD_GRAYSCALE),(img_width,img_height)))
                    for nn,_ in enumerate(os.listdir(val_img_dir_2 + 'z/'+ str(aug_dir) + '/')):
                        val_img_tmp.append(cv2.resize(cv2.imread(val_img_dir_2 + 'z/'+ str(aug_dir) + '/' + str(nn) + '.png',cv2.IMREAD_GRAYSCALE),(img_width,img_height)))
            
            
                        
            else:
                # train data set
                
                for aug_dir in aug_list:
                    os.chdir(label_dir+str(aug_dir)+'/')
                    for nn,_ in enumerate(os.listdir(label_dir + str(aug_dir) + '/')):
                        tmp_label.append(cv2.resize(cv2.imread(label_dir + str(aug_dir) + '/' + str(nn) + '.png',cv2.IMREAD_GRAYSCALE),(img_width,img_height)))
                    
                    train_data_name = str("box_num " + str(box_num) + " sub_box " + str(sub_num) + " peak_num " + str(peak_num) + " aug_dir " + str(aug_dir))
                    train_data_list.append(train_data_name)
                    
                    for nn,_ in enumerate(os.listdir(label_dir + 'z/'+ str(aug_dir) + '/')):
                        tmp_label.append(cv2.resize(cv2.imread(label_dir + 'z/'+ str(aug_dir) + '/'+ str(nn) + '.png',cv2.IMREAD_GRAYSCALE),(img_width,img_height)))
                    
                    train_data_name = str("box_num " + str(box_num) + " sub_box " + str(sub_num) + " peak_num " + str(peak_num) + " aug_dir " + str(aug_dir) + "with z")
                    train_data_list.append(train_data_name)


                    os.chdir(img_dir+str(aug_dir)+'/')
                    for nn,_ in enumerate(os.listdir(img_dir + str(aug_dir) + '/')):
                        tmp_img.append(cv2.resize(cv2.imread(img_dir + str(aug_dir) + '/' + str(nn) + '.png',cv2.IMREAD_GRAYSCALE),(img_width,img_height)))
                    for nn,_ in enumerate(os.listdir(img_dir + 'z/'+ str(aug_dir) + '/')):
                        tmp_img.append(cv2.resize(cv2.imread(img_dir + 'z/'+ str(aug_dir) + '/' + str(nn) + '.png',cv2.IMREAD_GRAYSCALE),(img_width,img_height)))


#%%

box_list = ['box01_add/','box02_add/']
#aug_list = [0,180]

label_dir_pre = '/home/swha/filament/data/label_smoothing/'
img_dir_pre = '/home/swha/filament/data/density_smoothing_plot_5_v2/'



for box_num in box_list:
    for sub_num in os.listdir(label_dir_pre + box_num):
        
        label_dir = label_dir_pre + box_num + sub_num + '/augmented/'
        img_dir = img_dir_pre + box_num + sub_num + '/'
        
            
        for aug_dir in aug_list:

            os.chdir(label_dir+str(aug_dir)+'/')
            for nn,_ in enumerate(os.listdir(label_dir + str(aug_dir) + '/')):
                tmp_label.append(cv2.resize(cv2.imread(label_dir + str(aug_dir) + '/' + str(nn) + '.png',cv2.IMREAD_GRAYSCALE),(img_width,img_height)))
            
            train_data_name = str("box_num " + str(box_num) + " sub_box " + str(sub_num) + " peak_num " + str(peak_num) + " aug_dir " + str(aug_dir))
            train_data_list.append(train_data_name)
            
            for nn,_ in enumerate(os.listdir(label_dir + 'z/'+ str(aug_dir) + '/')):
                tmp_label.append(cv2.resize(cv2.imread(label_dir + 'z/'+ str(aug_dir) + '/'+ str(nn) + '.png',cv2.IMREAD_GRAYSCALE),(img_width,img_height)))
            
            train_data_name = str("box_num " + str(box_num) + " sub_box " + str(sub_num) + " peak_num " + str(peak_num) + " aug_dir " + str(aug_dir) + "with z")
            train_data_list.append(train_data_name)


            os.chdir(img_dir+str(aug_dir)+'/')
            
            for nn,_ in enumerate(os.listdir(img_dir + str(aug_dir) + '/')):
                tmp_img.append(cv2.resize(cv2.imread(img_dir + str(aug_dir) + '/' + str(nn) + '.png',cv2.IMREAD_GRAYSCALE),(img_width,img_height)))
            for nn,_ in enumerate(os.listdir(img_dir + 'z/'+ str(aug_dir) + '/')):
                tmp_img.append(cv2.resize(cv2.imread(img_dir + 'z/'+ str(aug_dir) + '/' + str(nn) + '.png',cv2.IMREAD_GRAYSCALE),(img_width,img_height)))



#%%
train_data_list
#%%
val_data_list
# In[5]:
label_train = np.array(tmp_label)/255.
img_train = np.array(tmp_img)

label_val = np.array(val_label_tmp)/255.
img_val = np.array(val_img_tmp)


# In[6]:


img_train = img_train.reshape([-1,96,96,96])
label_train = label_train.reshape([-1,96,96,96])

img_val = img_val.reshape([-1,96,96,96])
label_val = label_val.reshape([-1,96,96,96])


# In[7]:


label_train = label_train.reshape([-1,img_depth,img_height,img_width])
img_train = img_train.reshape([-1,img_depth,img_height,img_width])

label_val = label_val.reshape([-1,img_depth,img_height,img_width])
img_val = img_val.reshape([-1,img_depth,img_height,img_width])

label_train = np.expand_dims(label_train,axis=-1)
img_train = np.expand_dims(img_train,axis=-1)

label_val = np.expand_dims(label_val,axis=-1)
img_val = np.expand_dims(img_val,axis=-1)


# In[21]:
i=1
j=23
plt.figure(figsize=[12,6])
plt.subplot(121)
plt.contourf(img_train[i,j,:,:,0])
plt.gray()
plt.subplot(122)
plt.contourf(label_train[i,j,:,:,0])
plt.gray()

#%%

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
run_name = '0213'
model_name = 'unet'

#model name = unet, resnet
os.chdir('/home/swha/filament/codes/')
if model_name == 'unet':
   from models.unet_models import *
   model = get_unet_v2()

if model_name == 'densenet':
    from models.DenseNet3D import *
    model = DenseNet3D_FCN((24, 96, 96, 1), nb_dense_block=2, growth_rate=8,
            nb_layers_per_block=2, upsampling_type='upsampling', classes=1, activation='sigmoid')
    model.compile(optimizer=Adam(lr=learning_rate, decay=decay_rate), loss=dice_coef_loss, metrics=[dice_coef])

ref_path = '/home/swha/filament/models/' + model_name + '/'

run_path = ref_path + run_name + '/'

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
with K.tf.device('/GPU:0'):
    hist = model.fit(img_train, label_train, batch_size=20, epochs=200,validation_data=(img_val,label_val),shuffle=True,
                    callbacks = [cb_checkpoint])



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

model.save_weights(run_path + 'model_' + run_name + '.h5')




# In[21]:


i = 61
j = 23
plt.figure(figsize=[16,8])
plt.subplot(121)
plt.contourf(img_val[i,j,:,:,0])
plt.subplot(122)
plt.contourf(label_val[i,j,:,:,0])
plt.gray()


# In[70]:


#img_val = img_file[:16,:,:,:,:]
save_dir = '/storage/filament/result/cluster_3d/40Mpc/prediction/aug_xyz_shuffle/'
# for i in range(test_re.shape[0]):
#     if not os.path.isdir(save_dir + str(i) +'/'):
#         os.makedirs(save_dir + str(i) +'/')
#     os.chdir(save_dir + str(i) +'/')
for j in range(test_re.shape[1]):
        plt.figure(figsize=[30,30])
        plt.imsave(str(j) +'.png',test_re[i,j,:,:],cmap='gray',dpi=300,format='png')


# In[ ]:


model.load_weights('/home/swha/filament/models/unet/0212/checkpoint_021292--0.4909_.hdf5')


# In[10]:


val = model.predict(img_val)
#%%

label_val_re = label_val.reshape([24,96,96,96])
val_re = val.reshape([24,96,96,96])
#%%
i = 2
j = 60
# plt.figure(figsize=[10,10])
# plt.imshow(img_val[i,:,:,0])
# plt.show()
plt.figure(figsize=[16,8])
plt.subplot(121)
plt.imshow(label_val_re[i,j,:,:])
plt.gray()
plt.subplot(122)
plt.imshow(val_re[i,j,:,:])
plt.gray()
plt.show()
# In[55]:

#%%
predict_path = '/home/swha/filament/data/prediction/0213/'

os.chdir(predict_path + 'label/')
j =  12
for i in range(96):
    plt.imsave(str(i) + '.png',label_val_re[j,i,:,:],cmap='gray',dpi=300,format='png')
os.chdir(predict_path + 'predict/')
for i in range(96):
    plt.imsave(str(i) + '.png',val_re[j,i,:,:],cmap='gray',dpi=300,format='png')
    
#%%
os.chdir('/storage/filament/result/cluster_3d/40Mpc/prediction/aug_xyz/pic/')
for i in range(16):
    for j in range(24):
        plt.figure(figsize=[30,30])
        plt.imsave('i_' + str(i) + ' j_' + str(j) +'.png',test[i,j,:,:,0],cmap='gray',dpi=300,format='png')


# In[ ]:



