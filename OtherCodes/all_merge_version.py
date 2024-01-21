from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer
from npy_append_array import NpyAppendArray
from math import pi
from torchvision import datasets, transforms
from sklearn.decomposition import PCA

import argparse
import torch
import pennylane as qml
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import time
import logging

# file_name=time.strftime("%d-%H%M%S", time.localtime())




def encode_circuit(qubits_num, section_number, x):
    # print(qubits_num,section_number,parameters)
    wires=[w for w in range(qubits_num)]
    # set H gates
    for wire in wires[0:-1]:
        qml.Hadamard(wire)
    # for each pixel set circuit
    for i in range(section_number):
        # print("here is inside the circuit",i,x[i])
        a=x[i]
        # print(i,a)
        i_b=str(np.binary_repr(i,width=qubits_num-1))[::-1]
        for pos,bit in enumerate(i_b):
            if bit=='0':
                qml.PauliX(pos)
        qml.ctrl(qml.RY, wires[0:-1], control_values=[1]*(qubits_num-1))(2*a, wires=qubits_num-1)
        for pos, bit in enumerate(i_b):
            if bit == '0':
                qml.PauliX(pos)
        qml.Barrier(wires)
    # print("encode done")

def ansatz_layer(W,qubits_num:int=7):
    # print("weights inside the circuit",W)
    for i in range(qubits_num):
        qml.Rot(W[i, 0], W[i, 1], W[i, 2], wires=i)
        if i !=qubits_num-1:
            qml.CNOT(wires=[i, i+1])
        else:
            qml.CNOT(wires=[i,0])

# @qml.qnode(dev_7, interface='autograd')
def FRQI_circuit(weights, x, qubits_n=7, section_n=64):
    # print("get w as ",weights)
    # print("input image in circuit is",x)
    encode_circuit(qubits_num=qubits_n,
                   section_number=section_n,
                   x=x)
    for W in weights:
        ansatz_layer(W)
    # result=[qml.counts(qml.PauliZ(i)) for i in range(qubits_n)]
    return qml.probs(qubits_n-1)

def variational_classifier(weights, bias, x,qnode):
    parity_result = cir(weights, x)
    return parity_result[0]+bias
    # return circuit(weights, x) + bias

def cost(weights, bias, X, Y,qnode):
    predictions = [variational_classifier(weights, bias, x,qnode) for x in X]
    # print("pre in cost is",predictions)
    return square_loss(Y, predictions)

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

def get_images(n_samples,r:int=8,c:int=8,batch_size:int=5):
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
    return train_loader,test_loader,rows*cols

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
        # sequentialize the image
        image_sequence = image.ravel()
        # pre processing
        for pos in range(len(image_sequence)):
            image_sequence[pos] = image_sequence[pos] * pi / 2
        # print("complete image is",image_sequence)
        sequences.append(image_sequence)
    sequences=np.array(sequences)
    return sequences

def image_preprocessing_angle(data:np.ndarray,qubits_number):
    sequences=[]
    for item in data:
        image=item[0]
        image_sequence = image.ravel()
        # pre processing
        sequences.append(image_sequence)
    sequences=np.array(sequences).T
    torchtensor = torch.as_tensor(sequences.T)
    pca = PCA(qubits_number)
    pca.fit(torchtensor)
    components = pca.transform(torchtensor)
    return components

def get_parameters():
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--method', type=str, help='FRQI/angle/amplitude', default="FRQI")
    parser.add_argument('--sample', type=int, help='number of samples', default=50)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.01)
    parser.add_argument('--batch', type=int, help='batch size', default=5)
    parser.add_argument('--epoch', type=int, help='epoch numbers', default=50)
    parser.add_argument('--layer', type=int, help='ansatz layer numbers', default=1)
    args = vars(parser.parse_args())
    return args

def logger_initialization(embedding_method:str):
    # set file name
    time_stamp = time.strftime("%d-%H%M%S", time.localtime())
    file_name = f"{embedding_method}_{time_stamp}"
    # set logging config
    logging.basicConfig(level=logging.INFO,
                        filename=f'./data/loggings/{file_name}.log')
    # logging file record parameter information
    return file_name

def record_save(epoch,
                file_name,
                start_time,
                weights,
                bias,
                epoch_acc,
                epoch_cost,
                epoch_acc_val
                ):
    record_item = np.array([[epoch_acc, epoch_cost, epoch_acc_val]])
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

def output_training_process(iter,acc,c):
    print(
        "Iter: {:d}| Accuracy: {:0.7f} | Loss: {}".format(
            iter, acc, c
        )
    )
    logging.info(f"Iter: {iter}| Accuracy: {acc} | Loss: {c}")

def FRQI_execute(embedding_methods,
                 sample_number,
                 batch_size,
                 learning_rate,
                 epoch_number,
                 layer_number):
    dev_7 = qml.device("default.qubit", wires=7, shots=1000)
    qnode = qml.QNode(FRQI_circuit, dev_7, interface='autograd')

    # print(embedding_methods, sample_number, batch_size, learning_rate, epoch_number, layer_number)
    # log the paramters
    file_name = logger_initialization(embedding_methods)
    logging.info(f"embedding methods {embedding_methods}, "
                 f"sample number {sample_number}, "
                 f"batch size {batch_size}, "
                 f"learning rate {learning_rate}, "
                 f"epoch number {epoch_number}, "
                 f"layer number {layer_number}")
    # load train and test data
    train_loader, test_loader, image_size = get_images(n_samples=sample_number,
                                                       batch_size=batch_size)
    # initialize weights and bias
    weights_init = 0.01 * np.random.randn(layer_number, 7, 3, requires_grad=True)
    bias_init = np.array(0.01, requires_grad=True)
    # initialize optimizer
    opt = NesterovMomentumOptimizer(learning_rate)
    weights = weights_init
    bias = bias_init
    # start training
    for epoch in range(epoch_number):
        start_time = time.time()
        print("start epoch {}".format(epoch + 1))
        logging.info(f"start epoch {epoch + 1}")
        predictions = []
        targets_train = []
        all_images = []
        iter = 1
        for batch_idx, (data, target) in enumerate(train_loader):
            encode_images_train = image_preprocessing(data)
            # print("old weights are ",weights)
            weights, bias, _, _ = opt.step(cost, weights, bias, encode_images_train, target,qnode)
            # print("new weights are ", weights)
            prediction = [variational_classifier(weights, bias, x,qnode) for x in encode_images_train]
            acc = accuracy(target, prediction)
            c = cost(weights, bias, encode_images_train, target)
            output_training_process(iter, acc, c)
            iter += 1
            # summary of data in this iteration
            for item in target:
                targets_train.append(item)
            for item in encode_images_train:
                all_images.append(item)
            for item in prediction:
                predictions.append(item)
            break
        # validation part
        targets_val = []
        encode_images_val = []
        for id_val, (data, target) in enumerate(test_loader):
            targets_val = target
            encode_images_val = image_preprocessing(data)
        predictions_val = [variational_classifier(weights, bias, x) for x in encode_images_val]
        epoch_acc = accuracy(targets_train, predictions)
        epoch_cost = cost(weights, bias, all_images, targets_train)
        epoch_acc_val = accuracy(targets_val, predictions_val)
        # save and store model
        record_save(epoch,
                    file_name,
                    start_time,
                    weights,
                    bias,
                    epoch_acc, epoch_cost, epoch_acc_val)

if __name__ == '__main__':
    # get paramters and file name
    args = get_parameters()
    embedding_methods = args['method']
    sample_number = args['sample']
    batch_size = args['batch']
    learning_rate = args['lr']
    epoch_number = args['epoch']
    layer_number = args['layer']

    if embedding_methods=="FRQI":
        FRQI_execute(embedding_methods,
                     sample_number,
                     batch_size,
                     learning_rate,
                     epoch_number,
                     layer_number)
    elif embedding_methods=="angle":
        pass


