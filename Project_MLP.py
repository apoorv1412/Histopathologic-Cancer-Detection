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

from sklearn.neural_network import MLPClassifier
import numpy as np

train_X = train_X[:, 32:64, 32:64, :]
test_X = test_X[:, 32:64, 32:64, :]

train_images = []
for i in range(len(train_X)):
    a = train_X[i]
    b = a.flatten()
    train_images.append(b)
train_images = np.array(train_images)

test_images = []
for i in range(len(test_X)):
    a = test_X[i]
    b = a.flatten()
    test_images.append(b)
test_images = np.array(test_images)

from sklearn import preprocessing
train_images = preprocessing.normalize(train_images)
test_images = preprocessing.normalize(test_images)

test_Y = test_Y.ravel()
train_Y = train_Y.ravel()

solver = 'relu'
nn = MLPClassifier(solver = 'lbfgs', max_iter = 100, hidden_layer_sizes = (200, 300, 100, 250, 300), verbose = True, activation = solver, alpha = 0.0001)
nn.fit(train_images, train_Y)
print ('solver = ', solver)
a = nn.predict(test_images)
print (nn.score(test_images, test_Y))
print (nn.score(train_images, train_Y))

solver = 'tanh'
nn = MLPClassifier(solver = 'lbfgs', max_iter = 100, hidden_layer_sizes = (300, 250, 200), verbose = True, activation = solver, alpha = 0.00001)
nn.fit(train_images, train_Y)
print ('solver = ', solver)
print (nn.score(test_images, test_Y))
print (nn.score(train_images, train_Y))

solver = 'relu'
nn = MLPClassifier(solver = 'lbfgs', max_iter = 100, hidden_layer_sizes = (300, 250, 200), verbose = True, activation = solver, alpha = 0.00001)
nn.fit(train_images, train_Y)
print ('solver = ', solver)
print (nn.score(test_images, test_Y))
print (nn.score(train_images, train_Y))

solver = 'identity'
nn = MLPClassifier(solver = 'lbfgs', max_iter = 100, hidden_layer_sizes = (300, 250, 200), verbose = True, activation = solver, alpha = 0.00001)
nn.fit(train_images, train_Y)
print ('solver = ', solver)
print (nn.score(test_images, test_Y))
print (nn.score(train_images, train_Y))
