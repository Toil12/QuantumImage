import time
import logging
from npy_append_array import NpyAppendArray

import autograd.builtins
from sklearn.decomposition import PCA
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer
import matplotlib.pyplot as plt
from math import pi
import torch
from torchvision import datasets, transforms
import pennylane as qml

QUBITS_NUMBER = 4

dev = qml.device("default.qubit", wires=QUBITS_NUMBER, shots=1000)
limits=[[-1.37332022,2.18239306],
 [-1.40379974,2.14600127],
 [-1.44892358,1.98171103],
 [-1.40922388,2.21770789],
 [-1.55775108,2.41359804],
 [-1.47722247,2.1297423 ],
 [-1.32422136,2.28240126],
 [-1.58264673,2.76999462],
 [-1.59794053,2.03189994],
 [-1.38024733,2.1177366 ],
 [-1.49680734,2.06916022],
 [-1.4323141 ,2.13822712],
 [-1.53581841,2.04166024],
 [-1.3885267 ,2.04384712]]


def encode_circuit_angle(a,qubits_number):
    for i in range(qubits_number):
        # angle=(a[i]-limits[i][0])/(limits[i][1]-limits[i][0])*np.pi
        angle=np.arctan(a[i])
        qml.RX(angle,wires=i)

def ansatz_layer(W,qubits_num):
    for i in range(qubits_num):
        qml.Rot(W[i, 0], W[i, 1], W[i, 2], wires=i)
        if i !=qubits_num-1:
            qml.CNOT(wires=[i, i+1])
        else:
            qml.CNOT(wires=[i,0])

@qml.qnode(dev, interface='autograd')
def circuit(weights, x,qubits_n):

    encode_circuit_angle(a=x,
                         qubits_number=qubits_n)
    for W in weights:
        ansatz_layer(W=W,qubits_num=qubits_n)
    # result=[qml.counts(qml.PauliZ(i)) for i in range(qubits_n)]
    return qml.probs(qubits_n-1)

def variational_classifier(weights, bias, x,qubits_number):
    parity_result = circuit(weights, x,qubits_number)
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
    predictions = [variational_classifier(weights, bias, f, QUBITS_NUMBER) for f in features]
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
                                                           transforms.Resize([rows, cols])
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

def image_preprocessing_angle(data:np.ndarray):
    sequences=[]
    for item in data:
        image=item[0]
        image_sequence = image.ravel()
        # pre processing
        sequences.append(image_sequence)
    sequences=np.array(sequences).T
    torchtensor = torch.as_tensor(sequences.T)
    pca = PCA(QUBITS_NUMBER)
    pca.fit(torchtensor)
    components = pca.transform(torchtensor)
    return components


if __name__ == '__main__':
    sample_number=100
    batch_size = 10
    learning_rate=0.01
    epoch_number=50
    layer_number = 2
    embedding_methods="angle"

    time_stamp = time.strftime("%d-%H%M%S", time.localtime())
    file_name = f"{embedding_methods}_{time_stamp}"
    logging.basicConfig(level=logging.INFO,
                        filename=f'./data/loggings/{file_name}')
    logging.info(f"embedding methods {embedding_methods}, "
                 f"sample number {sample_number}, "
                 f"batch size {batch_size}, "
                 f"learning rate {learning_rate}, "
                 f"epoch number {epoch_number}, "
                 f"layer number {layer_number}")
    train_loader, test_loader,image_size=get_images(n_samples=sample_number,
                                                    batch_size=batch_size)

    start_time=time.time()
    # initialize weights and bias

    weights_init = 0.01 * np.random.randn(layer_number, QUBITS_NUMBER, 3, requires_grad=True)
    bias_init = np.array(0.01, requires_grad=True)
    opt = NesterovMomentumOptimizer(learning_rate)
    weights = weights_init
    bias = bias_init

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
            encode_images = image_preprocessing_angle(data)
            weights, bias, _, _ = opt.step(cost, weights, bias, encode_images, target)

            prediction = [variational_classifier(weights, bias, x, QUBITS_NUMBER) for x in encode_images]
            acc = accuracy(target, prediction)
            c = cost(weights, bias, encode_images, target)
            print(
                "Iter: {}| Accuracy: {} | Loss: {}".format(
                    iter, acc, c
                )
            )
            logging.info(f"Iter: {iter}| Accuracy: {acc} | Loss: {c}")
            iter+=1
            for item in target:
                targets.append(item)
            for item in encode_images:
                all_images.append(item)
            for item in prediction:
                predictions.append(item)
        # validation part
        targets_val = []
        encode_images_val = []
        for id_val, (data, target) in enumerate(test_loader):
            targets_val = target
            encode_images_val = image_preprocessing_angle(data)
        predictions_val = [variational_classifier(weights, bias, x, QUBITS_NUMBER) for x in encode_images_val]

        epoch_acc = accuracy(targets, predictions)
        epoch_cost = cost(weights, bias, all_images, targets)
        epoch_acc_val = accuracy(targets_val, predictions_val)

        record_item=np.array([[epoch_acc,epoch_cost,epoch_acc_val]])
        end_time = time.time()
        time_cost = (end_time - start_time) / 60
        print("epoch {} : "
              "Accuracy {} , "
              "Loss {}, "
              "Time {}, "
              "Val Accuracy {}".format(epoch + 1, epoch_acc, epoch_cost, time_cost, epoch_acc_val))
        logging.info(f"epoch {epoch + 1} : "
                     f"Training Accuracy {epoch_acc} , "
                     f"Loss {epoch_cost}, "
                     f"Time {time_cost} minutes, "
                     f"Validation Accuracy {epoch_acc_val}")
        # print(record_item)
        with NpyAppendArray(f"./data/loss_epoch/{file_name}_loss.npy") as npaa:
            npaa.append(record_item)
        weights_store = np.array(weights).ravel()
        model = np.append(weights_store, bias)
        model = np.array([model])
        with NpyAppendArray(f"./data/model/{file_name}_model.npy") as npaa:
            npaa.append(model)