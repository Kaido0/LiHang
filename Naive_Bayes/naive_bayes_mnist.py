# -*- coding: utf-8 -*-
from __future__ import division
import pandas as pd
import numpy as np
import os
import time
from sklearn.model_selection import  train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

nb_classes=10
nb_featrues=784
'''
先验概率： 由于先验概率分母都是N，因此不用除于N，直接用分子即可。 
条件概率： 我们得到概率后再乘以10000，将概率映射到[0,10000]中，
          但是为防止出现概率值为0的情况，人为的加上1，使概率映射到[1,10001]中。
'''

def binaryzation(img):
    img=np.where(img>128,1,0)
    return img

def Train(train_set,train_label):
    prior_probability=np.zeros(nb_classes)
    conditional_probability=np.zeros([nb_classes,nb_featrues,2])

    for i in range(len(train_label)):
        img=binaryzation(train_set[i])
        label=train_label[i]
        for j in range(nb_featrues):
            conditional_probability[label,j,img[j]]+=1
            # lebel:0--->9
            # j:0--->748
            # img[j]:0,1
    for i in range(nb_classes):
        prior_probability[i] = (train_label == i).sum()
        print i, ":", prior_probability[i]
 #       pix_0=(conditional_probability[i,:,1]).sum()

        for j in range(nb_featrues):
            # 经过二值化后图像只有0，1两种取值
            pix_0=conditional_probability[i,j,0]
            pix_1=conditional_probability[i,j,1]

            # 计算0，1像素点对应的条件概率
            pro_0=pix_0/(pix_0+pix_1)*10000+1
            pro_1=pix_1/(pix_1+pix_0)*10000+1

            conditional_probability[i,j,0]=pro_0
            conditional_probability[i,j,1]=pro_1
    print "conditional_probability"
    print conditional_probability.shape
    return prior_probability,conditional_probability

def caculate_pro(image,label):
    pro=int(prior_probability[label])
    for i in range(len(image)):
        pro*=int(conditional_probability[label,i,image[i]])
    return pro

def predict(test_set,prior_probability,conditional_probability):
    predict=[]
    for img in test_set:
        img = binaryzation(img)
        max_label=0
        max_pro=caculate_pro(img,0)
        for i in range(1,10):
            prob=caculate_pro(img,i)

            if max_pro<prob:
                max_label=i
                max_pro=prob
        predict.append(max_label)

    return np.array(predict)



print 'Start read data'

time_1 = time.time()
data=pd.read_csv(os.getcwd()+'/train.csv')
data=data.values
image=data[:,1:]
label=data[:,0]

train_feature,test_feature,train_label,test_label=train_test_split(image,label, random_state=23323,test_size=0.7)
print "test_feature.shape：",test_feature.shape
print "train_feature.shape：",train_feature.shape

time_2 = time.time()
print 'read data cost ',time_2 - time_1,' second','\n'

print 'Start training'
prior_probability,conditional_probability = Train(train_feature,train_label)

time_3 = time.time()
print 'training cost ',time_3 - time_2,' second','\n'

print 'Start predicting'
test_predict = predict(test_feature, prior_probability, conditional_probability)
time_4 = time.time()
print 'predicting cost ', time_4 - time_3, ' second', '\n'

score = accuracy_score(test_label, test_predict)
print "The accruacy socre is ", score
