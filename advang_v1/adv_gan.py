#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from discriminator import discriminator
from generator import generator
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


# In[2]:


def preprocess():
    (x_train, y_train), (x_test, y_test)=tf.keras.datasets.cifar10.load_data()
    print("y_train_shape:",y_train.shape)
    y_train=y_train.reshape(50000)
    x_train_normalize =x_train.astype('float32') / 255.0
    x_test_normalize = x_test.astype('float32') / 255.0
    y_train_onehot=tf.one_hot( y_train,10)
    y_test_onehot=tf.one_hot( y_test,10)
    
    print("y_train_onehot_shape:",y_train_onehot.shape)
    
    #return x_train_normalize,x_test_normalize,y_train_onehot, y_test_onehot
    return x_train_normalize,x_test_normalize,y_train, y_test,y_train_onehot, y_test_onehot
x_train_normal,x_test_normal,y_train,y_test,y_train_onehot,y_text_onehot=preprocess()
attack_model=tf.keras.models.load_model("resnet_attack_model.h5")
x_train_normal=tf.convert_to_tensor(x_train_normal)
#model = tf.keras.models.load_model('g_net_1/')


# In[3]:


def loss_hing(predict,normal_pic):
    return tf.norm(predict-normal_pic)


# In[4]:


# dataset = tf.data.Dataset.from_tensor_slices({'a':test,'b':y_train_onehot}) 
# dataset = dataset.batch(100) 
# print(c['a'])
# for c in dataset:
#     print(c['a'])


# In[5]:


#d_net.weight


# In[6]:


def train(epoch=1,batch_size=256,wei1=0.7,wei2=0.8,wei3=-2):
    #train d_net
    d_net=discriminator()
    g_net=generator()
    d_net._name='out_d'
    
    d_net.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(0.0002,0.5),metrices=['accuracy'])


    #built all
    d_net.trainable=False
    attack_model._name='attack_model'   #reset model name for loss use
    attack_model.trainable=False
    inputs=tf.keras.layers.Input(shape=(32,32,3))

    noise=g_net(inputs)
    noise_picture=tf.keras.layers.add([noise,inputs],name='out_put_picture')  

    out_d=d_net(noise_picture)

    out_attack=attack_model(noise_picture)

    model_all=tf.keras.Model(inputs,[noise_picture,out_d,out_attack])
    model_all.compile(loss={'out_put_picture':loss_hing,'out_d':'categorical_crossentropy','attack_model':'categorical_crossentropy'}, optimizer='adam', metrics=['accuracy']
                     ,loss_weights={'out_put_picture':wei1,'out_d':wei2,'attack_model':wei3})


    dataset = tf.data.Dataset.from_tensor_slices({'img':x_train_normal,'label':y_train_onehot}) #set the deta use
    dataset=dataset.batch(batch_size)
    
    for _epoch in range(epoch):
        print("epoch =",_epoch,"-------------------------------")
        n=0
        d_loss_fake=0
        d_loss_real=0
        all_loss=0
        for batch_img in tqdm(dataset):
            n=n+1
            g_create=tf.math.add(g_net(batch_img['img']),batch_img['img'])  #create randon noise 
            #sampled_labels = np.random.randint(0, 10, (batch_size, 1))     #give the fake label randomly
            #g_create_lebal=tf.one_hot(sampled_labels,10)   
            
            #train d_net
            d_loss_fake+=d_net.train_on_batch(g_create,batch_img['label'])
            
            d_loss_real+=d_net.train_on_batch(batch_img['img'],batch_img['label'])
            
            #train g_net
            all_loss+=np.array(model_all.train_on_batch(batch_img['img'],{'out_put_picture':batch_img['img'],
                                                                'out_d':batch_img['label'],
                                                                'attack_model':batch_img['label']}))
        print("d_fake_loss",d_loss_fake/n)
        print("d_loss_real",d_loss_real/n)
        print("all_loss",all_loss/n)
        tf.keras.models.save_model(d_net,"d_net"+str(_epoch)+".h5")
        tf.keras.models.save_model(g_net,"g_net"+str(_epoch)+".h5")

    return g_net,d_net
            
            
            
    


# In[7]:


g_net,d_net=train(100,128,0.25,0.45,-1.2)


# plt.plot(d_acc)
# plt.plot(att_acc)
# plt.title('train History')
# plt.xlabel('Epoch')
# plt.legend(['D_acc','attack_acc'],loc='upper left')
# plt.show() -->

# 
# plt.plot(D_fake)
# plt.plot(d_real)
# plt.title('train History')
# plt.xlabel('Epoch')
# plt.legend(['D_fake','d_real'],loc='upper left')
# plt.show() -->

# In[33]:


def show_after(num):
    print("start label",y_train_onehot[num])
    row=x_train_normal.numpy()[num].copy()
    noi=g_net(row.reshape(1,32,32,3))
    plt.imshow(row.reshape(32,32,3))
    plt.show()
    print("noise=",noi)
    print("goal=",attack_model(row.reshape(1,32,32,3)+noi))


# In[36]:


def same_label(predict_label,label):  #notice input numpy array do not use tensor  
        return np.where(predict_label==predict_label.max())==np.where(label==label.max())


# def show_result(model,data=x_train_normal,label=y_train_onehot,g_net=g_net,show_num=10000):
#     num=0
#     success=0
#     pertubation=0
#     if show_num>x_train_normal.shape[0]:
#         print("error")
#         return 0
#     for a in tqdm(range(show_num)):
#         temp=model(data.numpy()[a].reshape(1,32,32,3)).numpy()
#         print("row class",temp)
#         if same_label(label[a].numpy().reshape(1,10),temp):
#             num+=1
#             temp1=model(data.numpy()[a].reshape(1,32,32,3)+g_net(data.numpy()[a].reshape(1,32,32,3))).numpy()
#             print("noise class",temp1)
#             print("dnet ",d_net(data.numpy()[a].reshape(1,32,32,3)+g_net(data.numpy()[a].reshape(1,32,32,3))))
#             if not same_label(label[a].numpy().reshape(1,10),temp1):
#                 print("row label give",label[a])
#                 pertubation+=tf.norm(g_net(data.numpy()[a].reshape(1,32,32,3)))/tf.norm(data.numpy()[a].reshape(1,32,32,3))
#                 print(pertubation)
#                 success+=1
#     print("rubboness=",pertubation/success)
#     print("num",num)
#     print("success",success)

# In[10]:


train(10,128,0.25,0.45,-1.2)


# In[22]:


tf.keras.utils.plot_model(model_all)


# In[ ]:




