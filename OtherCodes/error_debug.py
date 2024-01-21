import time


import autograd.builtins
from sklearn.decomposition import PCA
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer
import matplotlib.pyplot as plt
from math import pi
import torch
from torchvision import datasets, transforms
import pennylane as qml

dev = qml.device("default.qubit", wires=7,shots=1000)

def get_angles(x):


    return x

def statepreparation(a):
    qml.RY(a[0], wires=0)

    qml.CNOT(wires=[0, 1])
    qml.RY(a[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(a[2], wires=1)

    qml.PauliX(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RY(a[3], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(a[4], wires=1)
    qml.PauliX(wires=0)

def layer(W):

    qml.Rot(W[0, 0], W[0, 1], W[0, 2], wires=0)
    qml.Rot(W[1, 0], W[1, 1], W[1, 2], wires=1)
    qml.Rot(W[2, 0], W[2, 1], W[2, 2], wires=2)
    qml.Rot(W[3, 0], W[3, 1], W[3, 2], wires=3)

    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[3, 0])

@qml.qnode(dev, interface='autograd')
def circuit(weights, x):

    statepreparation(x)

    for W in weights:
        layer(W)

    return qml.expval(qml.PauliZ(0))

def variational_classifier(weights, bias, angles):
    return circuit(weights, angles) + bias

def accuracy(labels, predictions):

    loss = 0
    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            loss = loss + 1
    loss = loss / len(labels)

    return loss

def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2

    loss = loss / len(labels)
    return loss

def cost(weights, bias, features, labels):
    predictions = [variational_classifier(weights, bias, f) for f in features]
    return square_loss(labels, predictions)

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
                                                           ]))

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

def parameter_name(id):
    return "piexls_{}".format(str(id))

def recover_image(quantam_encode:list,show=True):
    image = quantam_encode
    recover_dict = {}
    for i in range(2 ** 6):
        recover_dict[i] = [0, 0]

    piexls_counts = 0
    for item in image.items():
        print(item)
        encode = item[0]
        count = int(item[1])
        pos = int(encode[1:],2)
        color = int(encode[0])
        piexls_counts += 1
        # print(pos,color)
        recover_dict[pos][color] += count

    image = np.zeros([8, 8])
    for key, val in recover_dict.items():
        raw = key // 8
        column = key % 8
        color_gray = val[1] / max((sum(val)),1)
        image[raw][column] = color_gray
    image=torch.tensor(image)
    if show:
        plt.imshow(image, cmap='gray')
        print(image)
        # plt.savefig('images/q.jpg')
        plt.show()

def show_recoverd_image(file_name):
    r=np.load(file_name)
    print(r)
    recover_image(r)

def image_preprocessing(data:np.ndarray,image_size:int=64):
    sequences=[]
    for image in data:
        # print(image)
        # print(image.shape)
        # fill circuit with angles
        # print(circ.parameters)

        # set angles ans parameters
        image_sequence = image.ravel()
        # pre processing
        for pos in range(len(image_sequence)):
            image_sequence[pos] = image_sequence[pos] * pi / 2
        # print("complete image is",image_sequence)
        sequences.append(image_sequence)
    sequences=np.array(sequences)
    return sequences


if __name__ == '__main__':
    sample_number=100
    train_loader, test_loader,image_size=get_images(n_samples=sample_number)
    train_images_q=np.empty([0],dtype=bool)
    train_loader_q=[]
    loss_record=[]

    start_time=time.time()
    # initialize weights and bias
    np.random.seed(0)
    num_qubits = 4
    num_layers = 2
    weights_init = weights_init = 0.01 * np.random.randn(num_layers, num_qubits, 3, requires_grad=True)
    bias_init = np.array(0.0, requires_grad=True)
    opt = NesterovMomentumOptimizer(0.001)
    weights = weights_init
    bias = bias_init
    batch_size=5

    for epoch in range(1):
        print("start epoch {}".format(epoch+1))
        predictions=[]
        targets=[]
        all_images=[]
        count = 0
        for batch_idx, (data, targets) in enumerate(train_loader):
            fig=plt.figure(figsize=(1,batch_size))
            images=[]
            for n in range(0,batch_size):
                fig.add_subplot(1, batch_size, n + 1)
                image=data[n][0]
                # plt.imshow(image, cmap='gray')
                image=image.ravel()
                images.append(image)
            # plt.show()
            images=np.array(images).T
            torchtensor = torch.as_tensor(images.T)
            pca=PCA(n_components=0.99)
            pca.fit(torchtensor)
            # print(pca.n_components_)
            components=pca.transform(torchtensor)
            # print(components.shape)
            print("components are: ",components)
            recovered=pca.inverse_transform(components)
            # print(recovered.shape)

            # fig2 = plt.figure(figsize=(1, batch_size))
            # for n in range(5):
            #     # original image size 28*28
            #     recover=torch.reshape(torch.as_tensor(recovered[n]),(28,28))
            #     fig2.add_subplot(1, batch_size, n + 1)
            #     plt.imshow(recover, cmap='gray')
            # plt.show()

            angles=get_angles(components)
            print(angles)




            break