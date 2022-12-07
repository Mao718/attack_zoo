#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv2D,BatchNormalization,Dense,Activation,add,GlobalAveragePooling2D
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


# In[2]:


def unit_block(before,filters):
    b=BatchNormalization(momentum=0.9, epsilon=1e-5)(before)
    b=Activation('relu')(b)
    b=Conv2D(filters,kernel_size=(3,3),strides=(1,1),padding='same')(b)
    b=BatchNormalization(momentum=0.9, epsilon=1e-5)(b)
    b=Activation('relu')(b)
    b=Conv2D(filters,kernel_size=(3,3),strides=(1,1),padding='same')(b)
    #print(before.shape)
    #print(b.shape)
    adds=add([before,b])
    return adds
def down_block(before,filters):
    
    b=BatchNormalization(momentum=0.9, epsilon=1e-5)(before)
    b=Activation('relu')(b)
    
    skip=Conv2D(filters,kernel_size=(3,3),strides=(2,2),padding='same')(b)
    
    b1=Conv2D(filters,kernel_size=(3,3),strides=(2,2),padding='same')(b)
    b1=BatchNormalization(momentum=0.9, epsilon=1e-5)(b1)
    b1=Activation('relu')(b1)
    b1=Conv2D(filters,kernel_size=(3,3),strides=(1,1),padding='same')(b1)
    
    adds=add([skip,b1])
    return adds


# In[3]:


def discriminator(): 
    inputs=Input(shape=(32,32,3))
    first=Conv2D(64,kernel_size=(3,3),strides=1,padding='same',name="first")(inputs)
    res1=unit_block(first,64)
    res2=unit_block(res1,64)

    down=down_block(res2,128)
    res3=unit_block(down,128)

    down2=down_block(res3,256)
    res4=unit_block(down2,256)

    down3=down_block(res4,512)
    res5=unit_block(down3,512)

    outbn=BatchNormalization(momentum=0.9, epsilon=1e-5)(res5)
    outre=Activation('relu')(outbn)

    out_1=GlobalAveragePooling2D()(outre)
    out_1= Dense(512,activation='relu',)(out_1)
    out_1=tf.keras.layers.Dropout(0.5)(out_1)
    out = Dense(10,activation='softmax',name='d_net_out')(out_1)

    output=out
    d_net=Model(inputs,output)
    return d_net


# In[ ]:




