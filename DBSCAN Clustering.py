#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


dataset=pd.read_csv("Mall_Customers.csv")


# In[3]:


dataset.head()


# In[5]:


dataset.shape


# In[6]:


X=dataset.iloc[:,[3,4]].values


# In[7]:


X


# In[8]:


from sklearn.cluster import DBSCAN


# In[10]:


dbscan=DBSCAN(eps=3,min_samples=4)


# In[12]:


model=dbscan.fit(X)


# In[13]:


labels=model.labels_


# In[14]:


from sklearn import metrics


# In[15]:


sample_cores=np.zeros_like(labels,dtype=bool)


# In[16]:


sample_cores


# In[18]:


sample_cores[dbscan.core_sample_indices_]=True


# In[19]:


sample_cores


# In[20]:


n_clusters=len(set(labels))- (1 if -1 in labels else 0)


# In[21]:


print(metrics.silhouette_score(X,labels))

