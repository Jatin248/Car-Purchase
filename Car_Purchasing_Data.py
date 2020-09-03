#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as  plt
import seaborn as sns


# In[23]:


car = pd.read_csv('Car_Purchasing_Data.csv',encoding='ISO-8859-1')


# In[24]:


car


# In[25]:


sns.pairplot(car)


# In[26]:


X = car.drop(['Customer Name','Customer e-mail','Country','Car Purchase Amount'],axis=1)


# In[27]:


y = car['Car Purchase Amount']


# In[28]:


from sklearn.preprocessing import MinMaxScaler
Scaler = MinMaxScaler()
X = Scaler.fit_transform(X)


# In[29]:


y=y.values.reshape(-1,1)


# In[30]:


y = Scaler.fit_transform(y)


# In[31]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)


# In[32]:


import tensorflow as tf


# In[33]:


model = tf.keras.models.Sequential()


# In[34]:


model.add(tf.keras.layers.Dense(units=30,activation='relu'))


# In[35]:


model.add(tf.keras.layers.Dense(units=30,activation='relu'))


# In[36]:


model.add(tf.keras.layers.Dense(units=1,activation='linear'))


# In[37]:


model.compile (optimizer='adam',loss='mean_squared_error')


# In[38]:


model.fit(X_train,y_train,epochs=50,batch_size=25,verbose=1,validation_split=0.2)


# In[39]:


X_test = np.array([[1,57,59000,5300,56000]])
y_predict = model.predict(X_test)


# In[40]:


print(y_predict)

