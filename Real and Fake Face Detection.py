#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
import os


# In[10]:


import sys
get_ipython().system('{sys.executable} -m pip install pandas')
get_ipython().system('{sys.executable} -m pip install opencv-python')


# ## Load Data

# In[4]:


fake = "/Users/apple/Desktop/Real-and-Fake-Face-Detection/archive/real_and_fake_face/training_fake"
real = "/Users/apple/Desktop/Real-and-Fake-Face-Detection/archive/real_and_fake_face/training_real"

real_path = os.listdir(real)
fake_path = os.listdir(fake)


# ## Load Image

# In[5]:


def load_img(path):
    image = cv2.imread(path)
    image = cv2.resize(image,(224, 224))
    return image[...,::-1]


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')


# ### take a glance at our real and fake image data

# In[7]:


print("real_path is a",type(real_path)," with",len(real_path),"images")
print("fake_path is a",type(fake_path)," with",len(fake_path),"images")


# ### access image with directory

# In[8]:


# access image
print(real_path[0])
print(real+"/"+real_path[0])


# In[9]:


plt.imshow(load_img(real+"/"+real_path[0]))


# ### Display some Real and Fake faces

# In[10]:


fig = plt.figure(figsize=(10, 10))

for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(load_img(real +"/"+ real_path[i]), cmap='gray')
    plt.suptitle("Real faces",fontsize=20)
    plt.axis('off')

plt.show()


# In[11]:


fig = plt.figure(figsize=(10,10))

for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(load_img(fake +"/"+ fake_path[i]), cmap='gray')
    plt.suptitle("Fakes faces",fontsize=20)
    plt.title(fake_path[i][:4])
    plt.axis('off')

plt.show()


# ### create train, validation and test data set

# In[12]:


for i in range(5):
    print(real_path[i])
for i in range(5):
    print(fake_path[i])


# In[ ]:




