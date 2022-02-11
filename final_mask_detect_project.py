#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install opencv-python')
import cv2
import matplotlib.pyplot as plt


# In[2]:


img = cv2.imread('neutral.png')
img.shape
img[0]


# In[3]:


plt.imshow(img)


# In[4]:


while True:
    cv2.imshow('result',img)
    if cv2.waitKey(2) == 27:
        break        
cv2.destroyAllWindows()


# In[5]:


haar_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
haar_data.detectMultiScale(img)


# In[6]:


while True:
    faces = haar_data.detectMultiScale(img)
    for x,y,w,h in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,255), 4)
    cv2.imshow('result',img)
    if cv2.waitKey(2) == 27:
        break
cv2.destroyAllWindows()


# In[9]:


capture = cv2.VideoCapture(0)
data = []

while True:
    flag, img = capture.read()
    if flag:
        faces = haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h), (255,0,255), 4)
            face = img[y:y+h, x:x+w, :]
            face = cv2.resize(face,(50,50))
            print(len(data))
            if len(data)<400:
                data.append(face)
        cv2.imshow('result',img)
        if cv2.waitKey(2) == 27 or len(data) >=200:
            break
            
capture.release()
cv2.destroyAllWindows()


# In[8]:


import numpy as np
np.save('without_mask.npy',data)


# In[10]:


np.save('with_mask.npy',data)


# In[11]:


plt.imshow(data[0])


# In[12]:


without_mask = np.load('without_mask.npy')
without_mask.shape


# In[13]:


with_mask = np.load('with_mask.npy')
with_mask.shape


# In[14]:


with_mask = with_mask.reshape(200,50*50*3)
without_mask = without_mask.reshape(200,50*50*3)


# In[15]:


with_mask.shape


# In[16]:


without_mask.shape


# In[17]:


X = np.r_[with_mask, without_mask]


# In[18]:


X.shape


# In[19]:


labels = np.zeros(X.shape[0])
labels[200:] = 1.0


# In[20]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# In[21]:


x_train, x_test, y_train, y_test = train_test_split(X,labels, test_size = 0.25)


# In[22]:


x_train.shape


# In[23]:


from sklearn.decomposition import PCA


# In[25]:


pca = PCA(n_components = 3)
x_train = pca.fit_transform(x_train)


# In[26]:


x_train[0]


# In[27]:


x_train.shape


# In[28]:


svm = SVC()
svm.fit(x_train,y_train)


# In[30]:


x_test = pca.transform(x_test)
y_pred = svm.predict(x_test)


# In[31]:


accuracy_score(y_test,y_pred)


# In[32]:


names = {0: 'Mask', 1: 'No Mask'}


# In[41]:


haar_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
capture = cv2.VideoCapture(0)
data = []
font = cv2.FONT_HERSHEY_COMPLEX

while True:
    flag, img = capture.read()
    if flag:
        faces = haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h), (255,0,255), 4)
            face = img[y:y+h, x:x+w, :]
            face = cv2.resize(face,(50,50))
            face = face.reshape(1,-1)
            face = pca.transform(face)
            pred = svm.predict(face)[0]
            n = names[int(pred)]
            cv2.putText(img,n,(x,y),font,1,(244,250,250),2)
            print(n)
        cv2.imshow('result',img)
        if cv2.waitKey(2) == 27:
            break
            
capture.release()
cv2.destroyAllWindows()


# In[ ]:




