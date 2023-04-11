# goal : how to make a residual connection ! 
# https://www.tensorflow.org/guide/keras/functional?hl=ko 굉장히 잘설명해되어있는것같다
# 이 링크를 보고 구현하였다 https://ganghee-lee.tistory.com/41

'''
기존 sequancial 방식에 한계점을 resnet을 구현하다 느끼고 이제부터 
모든 모델을 funcional 하게 설계해보았다.

resnet 에는 두가지 type에 residual block이 존재한다. 
identitiy block과 convolutional block 이 존재한다. 
'''

import numpy as np
import sys
sys.path.append(
    'C:\\Users\\drago\\university\\23.1 semester\\DL\\DLmodels-tensorflow')
print(sys.path)

from tensorflow import keras
from  tensorflow.keras.optimizers import Adam 
from readData import my_data_reader
from keras import models, layers
from keras import Input
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, initializers, regularizers, metrics
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import BatchNormalization, Conv2D, Activation, Dense, GlobalAveragePooling2D, MaxPooling2D, ZeroPadding2D, Add
 
import os
import matplotlib.pyplot as plt
import numpy as np
import math


#import data
data = my_data_reader.DataReader()
data.f_data_reader(rootfolder = "./data/RSP_data", img_size= 224)

#  Sequential 기존 방식의 한계이다. resnet 에 필요한 connection 을 만들어 주기 쉽지않다.

# number of classes
K = 3
input_tensor = (224, 224,3)

def identity_block(X, filters , kernel_size ,temp_bottelneck = False , preb_bottelneck = False):
    shortcut = X
    
    #optional to decrease kernels
    if preb_bottelneck == True : 
        X = Conv2D(kernel_size= (1,1), filters= filters*4 , padding = 'SAME')(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
    
    X = Conv2D(kernel_size= kernel_size, filters= filters , padding = 'SAME')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    X = Conv2D(kernel_size= kernel_size, filters= filters , padding = 'SAME')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    X = Conv2D( filters= filters, kernel_size= kernel_size, padding='SAME')(X)
    X = BatchNormalization()(X)
    
    #optional to increase kernels
    if temp_bottelneck == True : 
        X = Conv2D(kernel_size= (1,1) , filters= filters *4  , padding = 'SAME')(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        
        shortcut = Conv2D(kernel_size= (1,1) , filters= filters *4, padding = 'SAME')(shortcut)
        shortcut = BatchNormalization()(shortcut)
        shortcut = Activation('relu')(shortcut)
        
    X = Add()([X, shortcut])
    X = Activation('relu')(X)
    return X
    
def convolutional_block(X, filters , kernel_size ,temp_bottelneck = False , preb_bottelneck = False):
    shortcut = X
    
    #optional to decrease kernels
    if preb_bottelneck == True : 
        X = Conv2D(kernel_size= (1,1) , filters= filters *4, padding = 'SAME')(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
    
    X = Conv2D(kernel_size= kernel_size, filters= filters , padding = 'SAME')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    X = Conv2D(kernel_size= kernel_size, filters= filters , padding = 'SAME')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    X = Conv2D( filters= filters, kernel_size= kernel_size, padding='SAME')(X)
    X = BatchNormalization()(X)
    
    shortcut = Conv2D(filters= filters, kernel_size= kernel_size, padding = 'SAME')(shortcut)
    shortcut = BatchNormalization()(shortcut)
    
    #optional to increase kernels
    if temp_bottelneck == True : 
        X = Conv2D(kernel_size= (1,1) , filters= filters *4 , padding = 'SAME')(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)

        shortcut = Conv2D(kernel_size= (1,1) , filters= filters *4, padding = 'SAME')(shortcut)
        shortcut = BatchNormalization()(shortcut)
        shortcut = Activation('relu')(shortcut)
    
    X = Add()([X, shortcut])
    X = Activation('relu')(X)
    return X
    

def resnet50(input_shape, classes):
    X_input = Input(input_shape)
    X = X_input
    
    X = convolutional_block(X, 64, (3,3)) #conv
    X = identity_block(X, 64, (3,3))
    X = MaxPooling2D(2, 2, padding='SAME')(X) # not sure........
    
    X = convolutional_block(X, 128, (3,3)) #64->128, use conv block
    X = identity_block(X, 128, (3,3))
    X = MaxPooling2D(2, 2, padding='SAME')(X)
    
    X = convolutional_block(X, 256, (3,3)) #128->256, use conv block
    X = identity_block(X, 256, (3,3))
    X = MaxPooling2D(2, 2, padding='SAME')(X)
    
    X = convolutional_block(X, 512, (3,3)) #256->512, use conv block
    X = identity_block(X, 512, (3,3))
    X = MaxPooling2D(2, 2, padding='SAME')(X)
    
    X = GlobalAveragePooling2D()(X)
    
    # 이게 있어야 한다고 생각했는데 대부분의 구현 코드에 빠져있다..... 논문 봐야할듯
    # X = keras.layers.Flatten()(X)
    # X = Dense(1000, activation = 'relu')(X) 
    
    X = Dense(classes, activation = 'softmax')(X) # ouput layer 

    model =Model(inputs = X_input, outputs = X, name = "ResNet50")
    
    return model   

model = resnet50(input_shape= input_tensor , classes= K)
model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
model.summary()

EPOCH = 1
BATCH_SIZE = 128

# earlystopping = EarlyStopping(monitor='val_accuracy',
#                               patience=10, 
#                              )

# data = model.fit(X_train, 
#                  y_train, 
#                  validation_data=(X_valid, y_valid), 
#                  epochs=EPOCH, 
#                  batch_size=BATCH_SIZE, 
#                  callbacks=[ earlystopping],)
