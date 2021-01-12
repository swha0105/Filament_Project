#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
#import skimage.io as io
#import skimage.transform as trans
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler 
from keras import backend as keras
import keras.backend.tensorflow_backend as K
from keras import losses

## model 3D Unet
def get_unet_v1():
    
    n = 8
    
    inputs = Input((None, None,  None, 1))
    conv1 = Conv3D(n, (3, 3, 3), padding='same')(inputs)
    conv1 = BatchNormalization(axis=4)(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv3D(n, (3, 3, 3), padding='same')(conv1)
    conv1 = BatchNormalization(axis=4)(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv3D(n, (3, 3, 3), padding='same')(conv1)
    conv1 = BatchNormalization(axis=4)(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = Conv3D(n, (3, 3, 3), padding='same', strides = (2 ,2, 2))(conv1)
    pool1 = BatchNormalization(axis=4)(pool1)
    pool1 = Activation('relu')(pool1)
#    pool1 = Dropout(0.25)(pool1)
#    pool1 = MaxPooling3D(pool_size=(2, 2, 1))(conv1)

    conv22 = concatenate([pool1, MaxPooling3D(pool_size=(2, 2, 2))(inputs)], axis=4)
    conv2 = Conv3D(2*n, (3, 3, 3), padding='same')(conv22)
    conv2 = BatchNormalization(axis=4)(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv3D(2*n, (3, 3, 3), padding='same')(conv2)
    conv2 = BatchNormalization(axis=4)(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv3D(2*n, (3, 3, 3), padding='same')(conv2)
    conv2 = BatchNormalization(axis=4)(conv2)
    conv2 = Activation('relu')(conv2)                           
    pool2 = Conv3D(2*n, (3, 3, 3), padding='same', strides = (2, 2, 2))(conv2)
    pool2 = BatchNormalization(axis=4)(pool2)
    pool2 = Activation('relu')(pool2)
#    pool2 = Dropout(0.25)(pool2)
#    pool2 = MaxPooling3D(pool_size=(2, 2, 1))(conv2)

    conv33 = concatenate([pool2, MaxPooling3D(pool_size=(2, 2, 2))(conv22)], axis=4)
    conv3 = Conv3D(4*n, (3, 3, 3), padding='same')(conv33)    
    conv3 = BatchNormalization(axis=4)(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv3D(4*n, (3, 3, 3), padding='same')(conv3)    
    conv3 = BatchNormalization(axis=4)(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv3D(4*n, (3, 3, 3), padding='same')(conv3)
    conv3 = BatchNormalization(axis=4)(conv3)
    conv3 = Activation('relu')(conv3)
#    conv3 = Conv3D(4*n, (3, 3, 3), padding='same')(conv3)
#    conv3 = BatchNormalization(axis=4)(conv3)
#    conv3 = Activation('relu')(conv3)
    pool3 = Conv3D(2*n, (3, 3, 3), padding='same', strides = (2, 2, 2))(conv3)
    pool3 = BatchNormalization(axis=4)(pool3)
    pool3 = Activation('relu')(pool3)
    
    
    conv44 = concatenate([pool3, MaxPooling3D(pool_size=(2, 2, 2))(conv33)], axis=4)
    conv4 = Conv3D(8*n, (3, 3, 3), padding='same')(conv44)    
    conv4 = BatchNormalization(axis=4)(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv3D(8*n, (3, 3, 3), padding='same')(conv4)    
    conv4 = BatchNormalization(axis=4)(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv3D(8*n, (3, 3, 3), padding='same')(concatenate([conv44, conv4], axis=4))
    conv4 = BatchNormalization(axis=4)(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv3D(8*n, (3, 3, 3), padding='same')(conv4)    
    conv4 = BatchNormalization(axis=4)(conv4)
    conv4 = Activation('relu')(conv4)
#    conv3 = Dropout(0.5)(conv3)
#    pool3 = MaxPooling3D(pool_size=(2, 2, 1))(conv3)

#    conv4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool3)
#    conv4 = BatchNormalization(axis=1)(conv4)
#    conv4 = Activation('relu')(conv4)
#    conv4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv4)
#    conv4 = BatchNormalization(axis=1)(conv4)

    up5 = concatenate([Conv3DTranspose(2*n, (3, 3, 3), strides=(2, 2, 2), padding='same')(conv4), conv3], axis=4)
    conv5 = Conv3D(4*n, (3, 3, 3), padding='same')(up5)
    conv5 = BatchNormalization(axis=4)(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv3D(4*n, (3, 3, 3), padding='same')(conv5)
    conv5 = BatchNormalization(axis=4)(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv3D(4*n, (3, 3, 3), padding='same')(concatenate([up5, conv5], axis=4))
    conv5 = BatchNormalization(axis=4)(conv5)
    conv5 = Activation('relu')(conv5)
    
    up6 = concatenate([Conv3DTranspose(n, (3, 3, 3), strides=(2, 2, 2), padding='same')(conv5), conv2], axis=4)
    conv6 = Conv3D(2*n, (3, 3, 3), padding='same')(up6)
    conv6 = BatchNormalization(axis=4)(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv3D(2*n, (3, 3, 3), padding='same')(conv6)
    conv6 = BatchNormalization(axis=4)(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv3D(2*n, (3, 3, 3), padding='same')(concatenate([up6, conv6], axis=4))
    conv6 = BatchNormalization(axis=4)(conv6)
    conv6 = Activation('relu')(conv6)

    up7 = concatenate([Conv3DTranspose(n, (3, 3, 3), strides=(2, 2, 2), padding='same')(conv6), conv1], axis=4)
    conv7 = Conv3D(n, (3, 3, 3), padding='same')(up7)
    conv7 = BatchNormalization(axis=4)(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv3D(n, (3, 3, 3), padding='same')(conv7)
    conv7 = BatchNormalization(axis=4)(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv3D(n, (3, 3, 3), padding='same')(concatenate([up7, conv7], axis=4))
    conv7 = BatchNormalization(axis=4)(conv7)
    conv7 = Activation('relu')(conv7)
 #   up6 = concatenate([Conv3DTranspose(16, (2, 2, 1), strides=(2, 2, 1), padding='same')(conv5), conv1], axis=4)
 #   conv6 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(up6)
 #   conv6 = BatchNormalization(axis=1)(conv6)
 #   conv6 = Activation('relu')(conv6)
 #   conv6 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(conv6)
 #   conv6 = BatchNormalization(axis=1)(conv6)
 #   conv6 = Activation('relu')(conv6)
    conv8 = mo
    conv8 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv7)
    
    model = Model(inputs=[inputs], outputs=[conv8])
    
    learning_rate = 1e-3
    decay_rate = learning_rate/300
    
    model.compile(optimizer=Adam(lr=learning_rate, decay=decay_rate), loss='binary_crossentropy', metrics=['accuracy'])
#   model.compile(optimizer=Adam(lr=1e-5), loss=losses.mean_squared_error, metrics=[dice_coef])

    return model




def get_unet_v2():
    
    n = 8
    m = 3
    
    inputs = Input((None, None,  None, 1))
    conv1 = Conv3D(n, (m, m, m), padding='same')(inputs)
    conv1 = BatchNormalization(axis=4)(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv3D(n, (m, m, m), padding='same')(conv1)
    conv1 = BatchNormalization(axis=4)(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv3D(n, (m, m, m), padding='same')(conv1)
    conv1 = BatchNormalization(axis=4)(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = Conv3D(n, (m, m, m), padding='same', strides = (2 ,2, 2))(conv1)
    pool1 = BatchNormalization(axis=4)(pool1)
    pool1 = Activation('relu')(pool1)
#    pool1 = Dropout(0.25)(pool1)
#    pool1 = MaxPooling3D(pool_size=(2, 2, 1))(conv1)

    conv22 = concatenate([pool1, MaxPooling3D(pool_size=(2, 2, 2))(inputs)], axis=4)
    conv2 = Conv3D(2*n, (m, m, m), padding='same')(conv22)
    conv2 = BatchNormalization(axis=4)(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv3D(2*n, (m, m, m), padding='same')(conv2)
    conv2 = BatchNormalization(axis=4)(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv3D(2*n, (m, m, m), padding='same')(conv2)
    conv2 = BatchNormalization(axis=4)(conv2)
    conv2 = Activation('relu')(conv2)                           
    pool2 = Conv3D(2*n, (m, m, m), padding='same', strides = (2, 2, 2))(conv2)
    pool2 = BatchNormalization(axis=4)(pool2)
    pool2 = Activation('relu')(pool2)
    pool2 = Dropout(0.25)(pool2)
#    pool2 = MaxPooling3D(pool_size=(2, 2, 1))(conv2)

    conv33 = concatenate([pool2, MaxPooling3D(pool_size=(2, 2, 2))(conv22)], axis=4)
    conv3 = Conv3D(4*n, (m, m, m), padding='same')(conv33)    
    conv3 = BatchNormalization(axis=4)(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv3D(4*n, (m, m, m), padding='same')(conv3)    
    conv3 = BatchNormalization(axis=4)(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv3D(4*n, (m, m, m), padding='same')(conv3)
    conv3 = BatchNormalization(axis=4)(conv3)
    conv3 = Activation('relu')(conv3)
#    conv3 = Conv3D(4*n, (3, 3, 3), padding='same')(conv3)
#    conv3 = BatchNormalization(axis=4)(conv3)
#    conv3 = Activation('relu')(conv3)
    pool3 = Conv3D(2*n, (m, m, m), padding='same', strides = (2, 2, 2))(conv3)
    pool3 = BatchNormalization(axis=4)(pool3)
    pool3 = Activation('relu')(pool3)
    
    
    conv44 = concatenate([pool3, MaxPooling3D(pool_size=(2, 2, 2))(conv33)], axis=4)
    conv4 = Conv3D(8*n, (m, m, m), padding='same')(conv44)    
    conv4 = BatchNormalization(axis=4)(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv3D(8*n, (m, m, m), padding='same')(conv4)    
    conv4 = BatchNormalization(axis=4)(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv3D(8*n, (m, m, m), padding='same')(concatenate([conv44, conv4], axis=4))
    conv4 = BatchNormalization(axis=4)(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv3D(8*n, (m, m, m), padding='same')(conv4)    
    conv4 = BatchNormalization(axis=4)(conv4)
    conv4 = Activation('relu')(conv4)
    conv3 = Dropout(0.5)(conv3)
#    pool3 = MaxPooling3D(pool_size=(2, 2, 1))(conv3)

#    conv4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool3)
#    conv4 = BatchNormalization(axis=1)(conv4)
#    conv4 = Activation('relu')(conv4)
#    conv4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv4)
#    conv4 = BatchNormalization(axis=1)(conv4)

    up5 = concatenate([Conv3DTranspose(2*n, (m, m, m), strides=(2, 2, 2), padding='same')(conv4), conv3], axis=4)
    conv5 = Conv3D(4*n, (m, m, m), padding='same')(up5)
    conv5 = BatchNormalization(axis=4)(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv3D(4*n, (m, m, m), padding='same')(conv5)
    conv5 = BatchNormalization(axis=4)(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv3D(4*n, (m, m, m), padding='same')(concatenate([up5, conv5], axis=4))
    conv5 = BatchNormalization(axis=4)(conv5)
    conv5 = Activation('relu')(conv5)
    
    up6 = concatenate([Conv3DTranspose(n, (m, m, m), strides=(2, 2, 2), padding='same')(conv5), conv2], axis=4)
    conv6 = Conv3D(2*n, (m, m, m), padding='same')(up6)
    conv6 = BatchNormalization(axis=4)(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv3D(2*n, (m, m, m), padding='same')(conv6)
    conv6 = BatchNormalization(axis=4)(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv3D(2*n, (m, m, m), padding='same')(concatenate([up6, conv6], axis=4))
    conv6 = BatchNormalization(axis=4)(conv6)
    conv6 = Activation('relu')(conv6)

    up7 = concatenate([Conv3DTranspose(n, (m, m, m), strides=(2, 2, 2), padding='same')(conv6), conv1], axis=4)
    conv7 = Conv3D(n, (m, m, m), padding='same')(up7)
    conv7 = BatchNormalization(axis=4)(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv3D(n, (m, m, m), padding='same')(conv7)
    conv7 = BatchNormalization(axis=4)(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv3D(n, (m, m, m), padding='same')(concatenate([up7, conv7], axis=4))
    conv7 = BatchNormalization(axis=4)(conv7)
    conv7 = Activation('relu')(conv7)
 #   up6 = concatenate([Conv3DTranspose(16, (2, 2, 1), strides=(2, 2, 1), padding='same')(conv5), conv1], axis=4)
 #   conv6 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(up6)
 #   conv6 = BatchNormalization(axis=1)(conv6)
 #   conv6 = Activation('relu')(conv6)
 #   conv6 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(conv6)
 #   conv6 = BatchNormalization(axis=1)(conv6)
 #   conv6 = Activation('relu')(conv6)
    
    conv8 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv7)
    
    model = Model(inputs=[inputs], outputs=[conv8])
    
    learning_rate = 1e-3
    decay_rate = learning_rate/300
    
    model.compile(optimizer=Adam(lr=learning_rate, decay=decay_rate), loss=dice_coef_loss, metrics=[dice_coef])
#   model.compile(optimizer=Adam(lr=1e-5), loss=losses.mean_squared_error, metrics=[dice_coef])

    return model

