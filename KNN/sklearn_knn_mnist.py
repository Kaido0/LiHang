# -*- coding: utf-8 -*-
import pandas as pd
import time
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def get_hog_features(trainset):
    features=[]
    hog = cv2.HOGDescriptor('hog.xml')
    for img in trainset:
        img=np.reshape(img,(28,28))
        img_cv=img.astype('uint8')
        hog_feature=hog.compute(img_cv)
        features.append(hog_feature)
    feature=np.array(features)
    feature=np.reshape(feature,(-1,324))
    return feature



if __name__=='__main__':

    print 'Start read data'
    time_1 = time.time()

    data=pd.read_csv('train.csv')
    data=data.values
    imgs=data[:,1:]
    labels=data[:,0]
    features = get_hog_features(imgs)
    train_feature,test_feature,train_label,test_label=train_test_split(features,labels,test_size=0.1, random_state=23323)

    time_2 = time.time()
    print 'read data cost ',time_2 - time_1,' second','\n'

    print 'Start training'
    print 'knn do not need to train'
    neigh = KNeighborsClassifier(n_neighbors=10)
    neigh.fit(train_feature,train_label)

    time_3 = time.time()
    print 'training cost ', time_3 - time_2, ' second', '\n'

    print 'Start predicting'
    test_predict = neigh.predict(test_feature)
    time_4 = time.time()
    print 'predicting cost ', time_4 - time_3, ' second', '\n'

    score = accuracy_score(test_label, test_predict)
    print "The accruacy socre is ", score