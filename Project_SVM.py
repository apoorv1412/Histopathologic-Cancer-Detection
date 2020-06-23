'''
Reference: https://www.kaggle.com/aditya100/histopathologic-cancer-detection
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from skimage.io import imread
import keras.backend as k
import tensorflow as tf
import os
import cv2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

xx = 3
yy = 0
df = pd.DataFrame({'path': glob(os.path.join('../input/train', '*.tif'))})
df['id'] = df.path.map(lambda x: x.split('/')[(xx)].split(".")[yy])

lab = pd.read_csv('../input/train_labels.csv')
labels = lab
idd = "id"
df = df.merge(labels, on = idd)
p = 2000

df0 = df[df.label == 0].sample(p)
df1 = df[df.label == 1].sample(p)

df = pd.concat([df0, df1], ignore_index=True).reset_index()
df = df[["path", "id", "label"]]

df['image'] = df['path'].map(imread)

input_images = np.stack(list(df.image), axis = yy)
input_images.shape

train_Y = LabelBinarizer().fit_transform(df.label)
train_X = input_images
train_X, test_X, train_Y, test_Y = train_test_split(train_X, train_Y, test_size=0.0,shuffle = True)

train_Y = np.reshape(train_Y, (len(train_Y,)))
test_Y = np.reshape(test_Y, (len(test_Y,)))

import numpy as np

train_images = []
test_images = []

b = train_X[:,32:64,32:64,0]
b_ = test_X[:,32:64,32:64,0]

for i in range(len(train_X)):
    a = b[i]
    c = a.flatten()
    train_images.append(c)
train_images = np.array(train_images)

for i in range(len(test_X)):
    a = b_[i]
    c = a.flatten()
    test_images.append(c)
test_images = np.array(test_images)

from sklearn.svm import SVC

clf = SVC(kernel = 'linear', gamma = 0.00000001, C = 100, verbose = True)
clf.fit(train_images, train_Y)
print ('ok')
print (clf.score(train_images, train_Y))
print (clf.score(test_images, test_Y))


b = train_X[:,32:64,32:64,1]
b_ = test_X[:,32:64,32:64,1]

train_images = []
test_images = []

for i in range(len(train_X)):
    a = b[i]
    c = a.flatten()
    train_images.append(c)
train_images = np.array(train_images)

for i in range(len(test_X)):
    a = b_[i]
    c = a.flatten()
    test_images.append(c)
test_images = np.array(test_images)

from sklearn.svm import SVC

clf = SVC(kernel = 'linear', gamma = 0.00000001, C = 100, verbose = True)
clf.fit(train_images, train_Y)
print ('ok')
print (clf.score(train_images, train_Y))
print (clf.score(test_images, test_Y))

b = train_X[:,32:64,32:64,2]
b_ = test_X[:,32:64,32:64,2]

train_images = []
test_images = []

for i in range(len(train_X)):
    a = b[i]
    c = a.flatten()
    train_images.append(c)
train_images = np.array(train_images)

for i in range(len(test_X)):
    a = b_[i]
    c = a.flatten()
    test_images.append(c)
test_images = np.array(test_images)

from sklearn.svm import SVC

clf = SVC(kernel = 'linear', gamma = 0.00000001, C = 100, verbose = True)
clf.fit(train_images, train_Y)
print ('ok')
print (clf.score(train_images, train_Y))
print (clf.score(test_images, test_Y))

param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.001, 0.00001, 0.000005, 0.0000007], 
              'kernel': ['rbf']}  
  
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 
  
grid.fit(train_images, train_Y) 