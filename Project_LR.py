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

# %% [code]
# logisticRegr = LogisticRegression()
# logisticRegr.fit(train_X, train_Y)
train_X = np.einsum('abcd->adbc', train_X)
test_X = np.einsum('abcd->adbc', test_X)
# print(train_X.shape)
train_X = train_X[:,:,32:64,32:64]
test_X = test_X[:,:,32:64,32:64]
# print(train_X.shape)
train_images = []
test_images = []
for i in range(len(train_X)):
    a = train_X[i].flatten()
    train_images.append(a)
train_images = np.array(train_images)

for i in range(len(test_X)):
    a = test_X[i].flatten()
    test_images.append(a)
test_images = np.array(test_images)
logisticRegr = LogisticRegression(C = 0.001,penalty = 'l1')
logisticRegr.fit(train_images, train_Y)
score = logisticRegr.score(test_images, test_Y)
print(score)


# %% [code]
predictions = logisticRegr.predict_proba(test_images)
print(predictions)
# print(metrics.confusion_matrix(test_Y, predictions))

# %% [code]
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
# for i in range(2):
#     l2logreg.fit(x_train,q(i,y_train))
#     l2_test_accuracy.append(l2logreg.score(x_test,q(i,y_test)))
#     l2_train_accuracy.append(l2logreg.score(x_train,q(i,y_train))) 
#     y_score = l2logreg.fit(x_train, q(i,y_train)).decision_function(x_test)
#     fpr[i], tpr[i], _ = roc_curve(q(i,y_test),y_score)
#     plt.plot(fpr[i],tpr[i])
predictions = logisticRegr.predict_proba(test_images)
fpr, tpr, thresholds = roc_curve(test_Y, predictions[:,1])
plt.plot(fpr, tpr)
plt.show()