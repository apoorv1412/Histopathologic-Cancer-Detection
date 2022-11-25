'''
We have implemented using the PyTorch tutorial on CNNs
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

xx = 3
yy = 0
df = pd.DataFrame({'path': glob(os.path.join('../input/train', '*.tif'))})
df['id'] = df.path.map(lambda x: x.split('/')[(xx)].split(".")[yy])

lab = pd.read_csv('../input/train_labels.csv')
labels = lab
idd = "id"
df = df.merge(labels, on = idd)
p = 4000

df0 = df[df.label == 0].sample(p)
df1 = df[df.label == 1].sample(p)

df = pd.concat([df0, df1], ignore_index=True).reset_index()
df = df[["path", "id", "label"]]

df['image'] = df['path'].map(imread)

input_images = np.stack(list(df.image), axis = yy)
input_images.shape

train_Y = LabelBinarizer().fit_transform(df.label)
train_X = input_images
train_X, test_X, train_Y, test_Y = train_test_split(train_X, train_Y, test_size=2000, shuffle = True)
print(train_Y.shape)
print(train_X.shape)

train_X = np.einsum('abcd->adbc', train_X)
test_X = np.einsum('abcd->adbc', test_X)
print(train_X.shape)

train_X = train_X[:,:,32:64,32:64]
test_X = test_X[:,:,32:64,32:64]
plt.imshow(train_X[500][0], interpolation='nearest')
plt.show()
print(train_X.shape)
print(train_Y.shape)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn_conv2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.bn_fc1 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 15)
        self.fc3 = nn.Linear(15, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.bn_conv2(self.conv2(x))))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

import torchvision.models as models
resnet = models.resnet50(pretrained=True)
alexnet = models.alexnet(pretrained=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

train_Y = torch.from_numpy(train_Y)
for epoch in range(25): 
    
    combined = []
    for i in range(len(train_X)):
        combined.append([train_X[i], train_Y[i]])
    combined = np.array(combined)
    np.random.shuffle(combined)
    train_X = combined[:,0]
    train_Y = combined[:,1]

    running_loss = 0.0
        # get the inputs; data is a list of [inputs, labels]
    for i in range(len(train_X)):
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        a = np.reshape(train_X[i], (1,3,32,32))
        a = torch.from_numpy(a).to(device)
        outputs = net(a.float())
        loss = criterion(outputs, train_Y[i].to(device))
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
    print(running_loss)

print('Finished Training')


net = net.float()

correct = 0
total = 0
with torch.no_grad():
    max_prob = []
    y_true = []
    y_pred = []
    for i in range(len(train_X)):
        a = np.reshape(train_X[i], (1,3,32,32))
        a = torch.from_numpy(a)
        a = a.type('torch.DoubleTensor').to(device)
        labels = train_Y[i]
        outputs = net(a.float())

        probs = F.softmax(outputs, dim=1)
        probs = probs.cpu().numpy()
        max_prob.append(float(probs[0][1]))
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        y_true.append(int(labels))
        y_pred.append(int(predicted))
        if predicted==labels.to(device):
            correct+=1

train_Y = np.array(train_Y)
Y=[]
for i in range(len(train_Y)):
    Y.append(int(train_Y[i]))

max_prob = np.array(max_prob)
fpr, tpr, thresholds = roc_curve(y_true, max_prob)
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.savefig("p1.png")
plt.show()

print(confusion_matrix(y_true, y_pred))

print('Accuracy of the network on the 10000 training images: %d %%' % (
    100 * correct / total))

test_Y = torch.from_numpy(test_Y)

correct = 0
total = 0
with torch.no_grad():
    max_prob = []
    y_true = []
    y_pred = []
    for i in range(len(test_X)):
        a = np.reshape(test_X[i], (1,3,32,32))
        a = torch.from_numpy(a)
        a = a.type('torch.DoubleTensor').to(device)
        labels = test_Y[i]
        outputs = net(a.float())
        
        probs = F.softmax(outputs, dim=1)
        probs = probs.cpu().numpy()

        max_prob.append(float(probs[0][1]))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        y_true.append(int(labels))
        y_pred.append(int(predicted))
        if predicted==labels.to(device):
            correct+=1
            
test_Y = np.array(test_Y)
Y=[]
for i in range(len(test_Y)):
    Y.append(int(test_Y[i]))

max_prob = np.array(max_prob)
fpr, tpr, thresholds = roc_curve(y_true, max_prob)
plt.plot(fpr, tpr)
plt.title("ROC Curve: Test Set")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.savefig("p2.png")
plt.show()

print(confusion_matrix(y_true, y_pred))

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))