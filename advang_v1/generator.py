#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, Input, Dense, Reshape, Conv2DTranspose,   Activation, BatchNormalization, ReLU, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.datasets import cifar100
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


# In[2]:


def down_conv(x, filters, kernel_size, strides=2):
    x = Conv2D(filters=filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x
def up_conv(x, filters, kernel_size):
   x = Conv2DTranspose(filters=filters,
                       kernel_size=kernel_size,
                       strides=2,
                       padding='same')(x)
   x = BatchNormalization()(x)
   x = ReLU()(x)
   return x


# In[4]:


def generator():
   dae_inputs = Input(shape=(32, 32, 3), name='dae_input')
   down_conv1 = down_conv(dae_inputs, 32, 3)
   down_conv2 = down_conv(down_conv1, 64, 3)
   down_conv3 = down_conv(down_conv2, 128, 3)
   down_conv4 = down_conv(down_conv3, 256, 3)
   down_conv5 = down_conv(down_conv4, 256, 3, 1)

   up_conv1 = up_conv(down_conv5, 256, 3)
   merge1 = Concatenate()([up_conv1, down_conv3])
   up_conv2 = up_conv(merge1, 128, 3)
   merge2 = Concatenate()([up_conv2, down_conv2])
   up_conv3 = up_conv(merge2, 64, 3)
   merge3 = Concatenate()([up_conv3, down_conv1])
   up_conv4 = up_conv(merge3, 32, 3)

   final_deconv = Conv2DTranspose(filters=3,
                       kernel_size=3,
                       padding='same')(up_conv4)

   dae_outputs = Activation('sigmoid', name='dae_output')(final_deconv)
  
   return Model(dae_inputs, dae_outputs, name='g_net')


# In[ ]:




