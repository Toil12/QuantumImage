import time
import logging

import autograd.builtins
from sklearn.decomposition import PCA
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer
import matplotlib.pyplot as plt
from math import pi
import torch
from torchvision import datasets, transforms
import pennylane as qml

file_name=time.strftime("%d-%H%M%S", time.localtime())
logging.basicConfig(level=logging.INFO,
                    filename=f'./data/loggings/{file_name}')
dev = qml.device("default.qubit", wires=7,shots=1000)

def encode_circuit_angle(a,qubits_number:int=4):
    for i in range(qubits_number):
        qml.RX(a[i],wires=i)

def ansatz_layer(W,qubits_num:int=4):
    for i in range(qubits_num):
        qml.Rot(W[i, 0], W[i, 1], W[i, 2], wires=i)
        if i !=qubits_num-1:
            qml.CNOT(wires=[i, i+1])
        else:
            qml.CNOT(wires=[i,0])

@qml.qnode(dev, interface='autograd')
def circuit(weights, x,qubits_n:int=4):

    encode_circuit_angle(a=x,
                         qubits_number=qubits_n)
    for W in weights:
        ansatz_layer(W)
    # result=[qml.counts(qml.PauliZ(i)) for i in range(qubits_n)]
    return qml.probs(qubits_n-1)

def variational_classifier(weights, bias, x):
    parity_result = circuit(weights, x)
    return parity_result[0] + bias

def accuracy(labels, predictions):

    accuracy_count = 0
    # print(labels,predictions)
    for label, prediction in zip(labels, predictions):
        if abs(label-prediction) <= 0.5:
            accuracy_count = accuracy_count + 1
    accuracy = accuracy_count / len(labels)
    return accuracy

def square_loss(labels, predictions):
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
                                                           transforms.Resize([rows, cols])
                                                           # transforms.Normalize((0.1307,), (0.3081,)),
                                                           ]))

    # Leaving only labels 0 and 1
    idx = np.append(np.where(X_train.targets == 0)[0][:n_samples],
                    np.where(X_train.targets == 1)[0][:n_samples])

    X_train.data = X_train.data[idx]
    X_train.targets = X_train.targets[idx]

    train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size , shuffle=True)

# test data
    X_test = datasets.MNIST(root='./data', train=False, download=True,
                            transform=transforms.Compose([transforms.Grayscale(),
                                                          transforms.ToTensor(),
                                                          transforms.Resize([rows, cols])
                                                          ]))

    idx = np.append(np.where(X_test.targets == 0)[0][:n_samples],
                    np.where(X_test.targets == 1)[0][:n_samples])

    X_test.data = X_test.data[idx]
    X_test.targets = X_test.targets[idx]

    test_loader = torch.utils.data.DataLoader(X_test, batch_size=n_samples*2, shuffle=True)
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

def image_preprocessing(data:np.ndarray):
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
    sample_number=10000
    batch_size = 100
    learning_rate=0.01
    epoch_number=2
    layer_number = 2
    embedding_methods="angle"
    logging.info(f"embedding methods {embedding_methods}, "
                 f"sample number {sample_number}, "
                 f"batch size {batch_size}, "
                 f"learning rate {learning_rate}, "
                 f"epoch number {epoch_number}, "
                 f"layer number {layer_number}")
    train_loader, test_loader,image_size=get_images(n_samples=sample_number,
                                                    batch_size=batch_size)
    train_loader_q=[]
    loss_record=[]

    start_time=time.time()
    # initialize weights and bias

    weights_init = 0.01 * np.random.randn(layer_number, 7, 3, requires_grad=True)
    bias_init = np.array(0.01, requires_grad=True)
    opt = NesterovMomentumOptimizer(learning_rate)
    weights = weights_init
    bias = bias_init

    store=np.zeros([16,2])

    for epoch in range(epoch_number):
        start_time = time.time()
        print("start epoch {}".format(epoch+1))
        logging.info(f"start epoch {epoch + 1}")
        predictions=[]
        targets=[]
        all_images=[]
        count = 0
        iter=1

        for batch_idx, (data, target) in enumerate(train_loader):
            # print(data.shape)
            images=[]
            for n in range(0, len(data)):
                image=data[n][0]
                image=image.ravel()
                images.append(image)
            images=np.array(images).T
            torchtensor = torch.as_tensor(images.T)
            # print(torchtensor.shape)
            # 8
            pca=PCA(n_components=4)
            pca.fit(torchtensor)
            print(sum(pca.explained_variance_ratio_))
            # components=pca.transform(torchtensor)

        #     for i in range(components.shape[1]):
        #         store[i][0]=min(min(components[:][i]),store[i][0])
        #         store[i][1]=max(max(components[:][i]),store[i][1])
        # print(store)
