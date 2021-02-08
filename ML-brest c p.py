#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_csv("https://raw.githubusercontent.com/ingledarshan/AIML-B2/main/data.csv")


# In[4]:




df


# In[5]:


df.head()


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df['diagnosis'].unique()


# In[9]:


df.drop('id',axis=1,inplace=True )


# In[10]:


l = list(df.columns)
print(l)


# In[11]:


features_mean = l[1:11]
features_se = l[11:21]

features_worst = l[21:]


# In[12]:


df.head(2)


# In[13]:


df.head()


# In[14]:


df.shape


# In[15]:


df['diagnosis'].value_counts()


# In[16]:


df['diagnosis']=df['diagnosis'].map({"B":0,"M":1})


# In[17]:


df = df.drop("Unnamed: 32", axis=1)


# In[18]:


df["diagnosis"].unique()


# In[19]:


X = df.drop("diagnosis", axis=1)


# In[20]:


X.head()


# In[21]:


y=df["diagnosis"]


# In[29]:


y.head()


# In[30]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[31]:


df.shape


# In[32]:


X_train.shape


# In[33]:


X_test.shape


# In[34]:


y_train.shape


# In[35]:


y_test.shape


# In[36]:


X_train.head(1)


# In[37]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)


# In[38]:


X_train


# machine learning model

#  logistic regression

# In[41]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,y_train)


# In[44]:


LogisticRegression()


# In[45]:


y_pred=lr.predict(X_test)


# In[46]:


y_pred


# In[47]:


y_test


# In[51]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


# In[52]:


lr.acc=accuracy_score(y_test, y_pred)


# In[53]:


print(lr.acc)


# In[54]:


results = pd.DataFrame()
results


# In[56]:


tempResults = pd.DataFrame({'Algorithm':['Logistic Regression Method'], 'Accuracy':[lr.acc]})
results = pd.concat( [results, tempResults] )
results = results[['Algorithm','Accuracy']]
results


# # decision tree classifier

# In[58]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)


# In[61]:


y_pred = dt.predict(X_test)


# In[62]:


y_pred


# In[63]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


# In[64]:


dt.acc=accuracy_score(y_test, y_pred)


# In[65]:


tempResults = pd.DataFrame({'Algorithm':['Dicision tree classifier Method'], 'Accuracy':[dt.acc]})
results = pd.concat( [results, tempResults] )
results = results[['Algorithm','Accuracy']]
results


# # random forest classifier

# In[69]:


from sklearn.ensemble import  RandomForestClassifier
rfc= RandomForestClassifier()
rfc.fit(X_train,y_train)


# In[70]:


y_pred = rfc.predict(X_test)
y_pred


# In[71]:


rfc_acc = accuracy_score(y_test, y_pred)
print(rfc_acc)


# In[72]:


tempResults = pd.DataFrame({'Algorithm':['Random Forest Classifier Method'], 'Accuracy':[rfc_acc]})
results = pd.concat( [results, tempResults] )
results = results[['Algorithm','Accuracy']]
results


# # support vector machine
# 

# In[74]:


from sklearn import svm
svc = svm.SVC()
svc.fit(X_train,y_train)


# In[75]:


y_pred = svc.predict(X_test)
y_pred


# In[76]:


svc_acc = accuracy_score(y_test, y_pred)
print(svc_acc)


# In[77]:


tempResults = pd.DataFrame({'Algorithm':['Support Vector Classifier Method'], 'Accuracy':[svc_acc]})
results = pd.concat( [results, tempResults] )
results = results[['Algorithm','Accuracy']]
results


# # the end
