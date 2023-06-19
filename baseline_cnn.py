from pennylane.optimize import NesterovMomentumOptimizer
from math import pi
from torchvision import datasets, transforms
from net_simpleCNN import Net

import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import time
import numpy as np



def square_loss(labels, predictions):
    # print("labels and predicitons are ", labels, predictions)
    loss = 0
    lab=[]
    if type(labels)==list:
        for item in labels:
            lab.append(item.tolist())
    else:
        lab=labels
    # print(lab,predictions)
    for l, p in zip(lab,predictions):
        # print(l,p)
        loss = loss + (l - p) ** 2
    loss = loss / len(labels)
    # print(loss)
    return loss

def accuracy(labels, predictions):
    # print(labels)
    # print(predictions)
    # print("labels and predicitons in acc are ", labels, predictions)
    accuracy_count = 0
    # print(labels,predictions)
    for label, prediction in zip(labels, predictions):
        if abs(label-prediction) <= 0.5:
            accuracy_count = accuracy_count + 1
    accuracy = accuracy_count / len(labels)
    return accuracy

def cost(weights, bias, X, Y):
    predictions = [variational_classifier(weights, bias, x) for x in X]
    # print("pre in cost is",predictions)
    return square_loss(Y, predictions)

def get_images(n_samples:int=100,r:int=8,c:int=8,batch_size:int=5):
    # Concentrating on the first 100 samples
    n_samples = n_samples
    rows=r
    cols=c
# training data
    X_train = datasets.MNIST(root='./data', train=True, download=True,
                             transform=transforms.Compose([transforms.Grayscale(),
                                                           transforms.ToTensor(),
                                                           # transforms.Normalize((0.1307,), (0.3081,)),
                                                           transforms.Resize([rows, cols])]))

    # Leaving only labels 0 and 1
    idx = np.append(np.where(X_train.targets == 0)[0][:n_samples],
                    np.where(X_train.targets == 1)[0][:n_samples])

    X_train.data = X_train.data[idx]
    X_train.targets = X_train.targets[idx]

    train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size , shuffle=True)

# test data
    n_samples = 50

    X_test = datasets.MNIST(root='./data', train=False, download=True,
                            transform=transforms.Compose([transforms.Grayscale(),
                                                          transforms.ToTensor()]))

    idx = np.append(np.where(X_test.targets == 0)[0][:n_samples],
                    np.where(X_test.targets == 1)[0][:n_samples])

    X_test.data = X_test.data[idx]
    X_test.targets = X_test.targets[idx]

    test_loader = torch.utils.data.DataLoader(X_test, batch_size=1, shuffle=True)
    n_samples_show = 6
    return train_loader,test_loader,rows*cols

def image_preprocessing(data:np.ndarray,image_size:int=64):
    sequences=[]
    for image in data:
        # sequentialize the image
        image_sequence = image.ravel()
        # pre processing
        for pos in range(len(image_sequence)):
            image_sequence[pos] = image_sequence[pos] * pi / 2
        # print("complete image is",image_sequence)
        sequences.append(image_sequence)
    sequences=np.array(sequences)
    return sequences

if __name__ == '__main__':
    sample_number=50
    batch_size = 10
    learning_rate=0.01
    epoch_number=1
    train_loader, test_loader,image_size=get_images(n_samples=sample_number,
                                                    batch_size=batch_size)
    train_loader_q=[]
    loss_record=[]

    start_time=time.time()
    # initialize weights and bias
    layer_number=1
    opt = NesterovMomentumOptimizer(learning_rate)
    #start training
    for epoch in range(epoch_number):
        print("start epoch {}".format(epoch+1))
        predictions=[]
        targets=[]
        all_images=[]
        count = 0
        iter=1
        for batch_idx, (data, target) in enumerate(train_loader):
            print(data.shape)
            break
