
# coding: utf-8

# In[24]:


# correct code processor
import numpy as np
import cv2 as cv
import csv
from random import shuffle
a = []
def preprocessing(path,path_csv,no_of_img,no_of_fliped):#'C:/Users/SANKET/Desktop/dataset_sedanvssuv.csv'
    i=1
    b=[]
    counter = 0
    c = 0
    with open(path_csv) as file:
        reader = csv.reader(file)
        for f in reader:
            img_obj =( cv.imread(path+'/'+str(i).zfill(3)+'.jpg'))/(255*50)
#             print(img_obj)
            i = i+1
            a.append([img_obj,f[1]])
            c = c+1
            if counter<no_of_fliped:
                j = np.random.choice(no_of_img,1)
                if j not in b:
                    b.append(j)
                    img_obj_1 = cv.imread(path+'/'+str(j[0]).zfill(3)+'.jpg')
                    img_obj_1 = ((img_obj_1)/(255*50))
                    flip_img = cv.flip(img_obj_1,1)
                    a.append([flip_img,f[1]])
                    counter = counter + 1
    shuffle(a)
#     print(len(a))
#     print(counter)
#     print(c)
    return a 
# preprocessing('C:/Users/SANKET/Desktop/classifier/trial_dataset','C:/Users/SANKET/Desktop/classifier/trial_sedanvssuv.csv',240,100)


# In[25]:


print(a)


# In[26]:


import os
# count = 0
i = 0
def batch_formation(batch_size,List,path_all):
    reshaped_array = np.empty((batch_size,180000))
    label_array = np.empty((batch_size,1))
    os.chdir(path_all)  #'C:/Users/SANKET/Desktop/1'
    k = 0
    count = 0
    global i
    for _ in range(batch_size):
        f = List[count+i] #i baki hai
        count = count + 1
        i = i + 1
        if i > 400:
            i = 0
        img = f[0].reshape(1,180000)
        if k < batch_size:
            reshaped_array[k] = img
            label_array[k]= f[1]
            k = k+1
    return reshaped_array,label_array

# x,y = batch_formation(20,preprocessing('C:/Users/SANKET/Desktop/1','C:/Users/SANKET/Desktop/dataset_sedanvssuv.csv'),'C:/Users/SANKET/Desktop/1')
# print(x)


# In[27]:


def weight_biases(batch_size):
    w = np.random.normal(0.0,1.0,(180000,1))
    b = np.random.normal(0.0,1.0,(batch_size,1))
#     print(w,'\n')
#     print(b)
#     print(w.shape)
    return w,b
# weight_biases(20)


# In[28]:


def forward_pass(w,b,x):
    import numpy as np
#     (w,b) = weight_biases()
#     (x,m)= batch_formation(20,preprocessing('C:/Users/SANKET/Desktop/1'))
    y = np.add(np.dot(x,w),b)
#     matmul use karna hai
#     print(np.shape(y))
#     print(y)
    return y
# forward_pass()


# In[29]:


import numpy as np
def sigmoid(x,batch_size):
    y = np.empty([batch_size,1])
#     g = np.empty([20,1])
#     k = 0
    g = 1/(1+np.exp(-x))
    for i in range(batch_size) :
        if g[i] >= 0.5:
            y[i] = 1
        else:
            y[i] = 0
    return g,y
# sigmoid(forward_pass(),20)


# In[30]:


import numpy as np
def cost_function(batch_size,w,b,path):
#     f = preprocessing('C:/Users/SANKET/Desktop/1','C:/Users/SANKET/Desktop/dataset_sedanvssuv.csv')
    (x,y) = batch_formation(batch_size,a,path)
#     (w,b) = weight_biases(batch_size,w_new,b_new)
    (h,t) = sigmoid(forward_pass(w,b,x),batch_size)
    A = -(y*np.log(h)+(1-y)*np.log(h))
    cost_total = np.sum(A)
    cost_avg = cost_total/(batch_size)
#     print(A,cost_total,cost_avg)
#     print(A.shape)
    return A,cost_total,cost_avg
# cost_function(20)


# In[31]:


import numpy as np
def gradient_descent(m,A,batch_size,w,b,path):
#     (w,b) = weight_biases(batch_size<w_new,b_new)
#     f = preprocessing('C:/Users/SANKET/Desktop/1','C:/Users/SANKET/Desktop/dataset_sedanvssuv.csv')
    (x,y)= batch_formation(batch_size,a,path)
    (g) = forward_pass(w,b,x)
    (h,e) = sigmoid(g,batch_size)
    w = w - (A/m)*np.matmul(np.transpose(x),(h-y))
    b = b - (A/m)*(h-y)
#     print(np.shape(w),np.shape(b))
    return w,b
# gradient_descent(100,0.0001) 


# In[57]:


def accuracy(batch_size,w,b,path):
    count = 0
#     (w,b) = weight_biases(batch_size,w_new,b_new)
#     temp_2 = preprocessing('C:/Users/SANKET/Desktop/1','C:/Users/SANKET/Desktop/dataset_sedanvssuv.csv')
    (e,f) = batch_formation(batch_size,a,path)
#     (batch1,lable2) = batch_formation(20,temp_2,'C:/Users/SANKET/Desktop/1')
    (h,y) = sigmoid(forward_pass(w,b,e),batch_size)
    temp = f-y
    for i in range(20):
        if temp[i]!=0:
            count = count +1
    s = (1-(count/20))*100
#     print(s)
    return s
# accuracy()


# In[33]:


wnew,bnew = weight_biases(20)


# In[46]:


accu = []
def train_it(w_new,b_new,batch_size,alpha,listy,path):
    for _ in range(int((len(listy)/batch_size))):
        (batch,lable) = batch_formation(batch_size,listy,path)
        (A,cost_total,cost_avg) = cost_function(batch_size,w_new,b_new,path)
        (w_new,b_new) = gradient_descent(400,alpha,batch_size,w_new,b_new,path)
        acc = accuracy(batch_size,w_new,b_new,path)
        return w_new,b_new,cost_total,acc
# train_it(w_new,b_new,400)


# In[52]:


# validation ka code
def validation(v_w_new,v_b_new,v_batch_size,v_alpha,v_listy,path):
    for _ in range(int((len(v_listy)/v_batch_size))):
        (v_batch,v_lable) = batch_formation(v_batch_size,v_listy,path)
        (A,cost_total,cost_avg) = cost_function(v_batch_size,v_w_new,v_b_new,path)
        (w_new,b_new) = gradient_descent(400,v_alpha,v_batch_size,v_w_new,v_b_new,path)
        acc = accuracy(v_batch_size,v_w_new,v_b_new,path)
        return v_w_new,v_b_new,cost_total,acc


# In[58]:


# wnew,bnew = weight_biases(20)
LList = preprocessing('C:/Users/SANKET/Desktop/classifier/trial_dataset','C:/Users/SANKET/Desktop/classifier/trial_sedanvssuv.csv',240,100)
v_LList = preprocessing('C:/Users/SANKET/Desktop/classifier/validation_dataset','C:/Users/SANKET/Desktop/classifier/validation_sedanvssuv.csv',40,20)
for i in range(40):
    (x,y,z,t) = train_it(wnew,bnew,20,0.05,LList,'C:/Users/SANKET/Desktop/classifier/trial_dataset')
    (v_x,v_y,v_z,v_t) = validation(wnew,bnew,20,0.05,v_LList,'C:/Users/SANKET/Desktop/classifier/validation_dataset')
    print('for train',z,t)
    print('for validation',v_z,v_t)
# accu_avg = np.sum(np.array(accu))/30
# print('accuracy average = ',accu_avg)

