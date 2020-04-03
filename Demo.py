#!/usr/bin/env python
# coding: utf-8

# In[1]:
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

get_ipython().run_line_magic('matplotlib', 'tk')
from pylab import *
import cv2


# In[2]:


rcParams['figure.figsize'] = 10, 10


# In[3]:


from dataset import load_image


# In[4]:


import torch


# In[5]:


from utils import cuda


# In[6]:


from generate_masks import get_model


# In[7]:


from albumentations import Compose, Normalize


# In[8]:


"from albumentations.torch.functional import img_to_tensor"


# In[9]:

import torchvision.transforms.functional as F


def img_to_tensor(im, normalize=None):

    tensor = torch.from_numpy(np.moveaxis(im / (255. if im.dtype == np.uint8 else 1), -1, 0).astype(np.float32))
    if normalize is not None:
        return F.normalize(tensor, **normalize)
    return tensor


def img_transform(p=1):
    return Compose([
        Normalize(p=1)
    ], p=p)


# In[10]:


def mask_overlay(image, mask, color=(0, 255, 0)):
    """
    Helper function to visualize mask on the top of the car
    """
    mask = np.dstack((mask, mask, mask)) * np.array(color)
    mask = mask.astype(np.uint8)
    weighted_sum = cv2.addWeighted(mask, 0.5, image, 0.5, 0.)
    img = image.copy()
    ind = mask[:, :, 1] > 0    
    img[ind] = weighted_sum[ind]    
    return img


# In[11]:


model_path = 'unet11_binary_20/model_0.pt'
model = get_model(model_path, model_type='UNet11', problem_type='binary')


# In[12]:


img_file_name = 'data/123.png'
"gt_file_name = 'data/cropped_train/instrument_dataset_3/binary_masks/frame004.png'"


# In[13]:


image = load_image(img_file_name)
"gt = cv2.imread(gt_file_name, 0) > 0"


# In[14]:



imshow(image)


# In[15]:


with torch.no_grad():
    input_image = torch.unsqueeze(img_to_tensor(img_transform(p=1)(image=image)['image']).cuda(), dim=0)


# In[16]:

mask = model(input_image)

# In[17]:


mask_array = mask.data[0].cpu().numpy()[0]


# In[18]:


plt.imshow(mask_array > 0)


# In[19]:


imshow(mask_overlay(image, (mask_array > 0).astype(np.uint8)))

