import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K

def IoU_loss(y_true,y_pred):
    smooth = 1e-12
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(1-jac)


def unet(pretrained_weights = None,input_size = (712,1072,1)):
    inputs = Input(input_size)
    
    D1conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    D1conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(D1conv1)
    D1pool = MaxPooling2D(pool_size=(2, 2))(D1conv2)
    
    D2conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(D1pool)
    D2conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(D2conv1)
    D2pool = MaxPooling2D(pool_size=(2, 2))(D2conv2)
    
    D3conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(D2pool)
    D3conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(D3conv1)
    D3pool = MaxPooling2D(pool_size=(2, 2))(D3conv2)
    
    D4conv1 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(D3pool)
    D4conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(D4conv1)
    D4pool = MaxPooling2D(pool_size=(2, 2))(D4conv2)
    
    D5conv1 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(D4pool)
    D5conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(D5conv1)
    D5pool = MaxPooling2D(pool_size=(2, 2))(D5conv2)
    
    D6conv1 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(D5pool)
    D6conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(D6conv1)
    
    U1up = UpSampling2D(size = (2,2))(D6conv2)
    U1merge = concatenate([D5conv2,U1up], axis = 3)
    U1conv1 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(U1merge)
    U1conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(U1conv1)
    
    U2up = UpSampling2D(size = (2,2))(U1conv2)
    U2merge = concatenate([D4conv2,U2up], axis = 3)
    U2conv1 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(U2merge)
    U2conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(U2conv1)
    
    U3up = UpSampling2D(size = (2,2))(U2conv2)
    U3merge = concatenate([D3conv2,U3up], axis = 3)
    U3conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(U3merge)
    U3conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(U3conv1)
    
    U4up = UpSampling2D(size = (2,2))(U3conv2)
    U4merge = concatenate([D2conv2,U4up], axis = 3)
    U4conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(U4merge)
    U4conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(U4conv1)
    
    U5up = UpSampling2D(size = (2,2))(U4conv2)
    U5merge = concatenate([D1conv2,U5up], axis = 3)
    U5conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(U5merge)
    U5conv2 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(U5conv1)    

    S = Conv2D(1, 1, activation = 'sigmoid')(U5conv2)

    model = Model(input = inputs, output = S)

    model.compile(optimizer = Adam(lr = 1e-4), loss = IoU_loss, metrics = ['accuracy'])
    
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model