#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 


# In[3]:


dataset = pd.read_csv("STOCK_WITH_MISSING.csv")


# In[4]:


dataset


# In[5]:


dataset.isna().sum()


# In[6]:


cols = ['Date','Adj_Close']
dataset = dataset.drop(cols, axis=1)


# In[7]:


dataset


# In[12]:


X = dataset.iloc[:, :-1].values
X


# In[13]:


Y = dataset.iloc[:, -1].values 
Y


# In[14]:


dataset.describe()


# In[16]:


X[:,0:1]


# In[17]:


from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer=imputer.fit(X[:,0:4])
X[:,0:4]=imputer.transform(X[:,0:4])
X


# In[18]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=1)


# In[21]:


print(X_test)


# In[20]:


print(X_train)


# In[23]:


scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)
print(X_train)
print(X_test)


# In[ ]:




