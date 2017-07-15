# -*- coding: utf-8 -*-

import cv2
import time
import logging
import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree


total_class = 10

def binaryzation(img):
    cv_img = img.astype(np.uint8)
    cv2.threshold(cv_img,50,1,cv2.THRESH_BINARY_INV,cv_img)
    # gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    return cv_img

def binaryzation_features(trainset):
    features = []

    for img in trainset:
        img = np.reshape(img,(28,28))
        cv_img = img.astype(np.uint8)

        img_b = binaryzation(cv_img)
        # hog_feature = np.transpose(hog_feature)
        features.append(img_b)

    features = np.array(features)
    features = np.reshape(features,(-1,784))

    return features


if __name__ == '__main__':


    print 'Start read data'
    time_1 = time.time()
    raw_data = pd.read_csv('train.csv')
    data = raw_data.values

    imgs = data[:,1:]
    labels = data[:,0]

    # 图片二值化
    features = binaryzation_features(imgs)
    train_features, val_features, train_labels, val_labels = train_test_split(features, labels, test_size=0.33,random_state=23323)

    time_2 = time.time()
    print 'read data cost ', time_2 - time_1, ' second', '\n'

    print 'Start training'

    clf=tree.DecisionTreeClassifier()
    clf.fit(train_features,train_labels)
    time_3 = time.time()
    print 'training cost ', time_3 - time_2, ' second', '\n'


    print 'Start predicting'
    val_pre = clf.predict(val_features)
    time_4 = time.time()
    print 'predicting cost ', time_4 - time_3, ' second', '\n'

    score = accuracy_score(val_labels, val_pre)
    print "The accruacy socre is ", score

    test = pd.read_csv('test.csv')
    test = test.values
    test_feature = binaryzation_features(test)
    test_pre = clf.predict(test_feature)


    out_file = open("prediction.csv", "w")
    out_file.write("ImageId,Label\n")
    for i in range(len(test_pre)):
        out_file.write(str(i + 1) + "," + str(int(test_pre[i])) + "\n")
    out_file.close()

